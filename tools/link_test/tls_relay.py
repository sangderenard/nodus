"""TEST ONLY: a TLS front for link_test's wss cases, standing where Apache stands in production (TLS terminated in front of
the native LinkListener on 127.0.0.1). It makes a fresh self-signed certificate for 127.0.0.1, prints one line

    RELAY <port> <sha256 of the certificate DER, hex>

and relays every TLS connection to 127.0.0.1:<target-port> byte for byte until stdin closes. Never part of a link's data
path outside this test (research case 98d49cb8520e44d092c5d134c37c3c32).

    python tls_relay.py <target-port> <scratch-dir>
"""
import asyncio
import datetime
import hashlib
import ipaddress
import ssl
import sys
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


def certificate(directory: Path):
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(hours=1))
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
            .sign(key, hashes.SHA256()))
    cert_path, key_path = directory / "relay_cert.pem", directory / "relay_key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    return cert_path, key_path, hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()


async def pump(reader, writer):
    try:
        while data := await reader.read(1 << 16):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, ssl.SSLError):
        pass
    finally:
        writer.close()


async def main(target: int, directory: Path):
    cert_path, key_path, digest = certificate(directory)
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(cert_path, key_path)

    async def accept(reader, writer):
        try:
            up_reader, up_writer = await asyncio.open_connection("127.0.0.1", target)
        except OSError:
            writer.close()
            return
        await asyncio.gather(pump(reader, up_writer), pump(up_reader, writer))

    server = await asyncio.start_server(accept, "127.0.0.1", 0, ssl=context)
    print(f"RELAY {server.sockets[0].getsockname()[1]} {digest}", flush=True)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, sys.stdin.read)               # the test closes stdin to stop the relay
    server.close()


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]), Path(sys.argv[2])))
