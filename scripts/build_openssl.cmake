# OpenSSL, pinned: the release tarball, checked against its SHA-256, built in isolation and installed into
# dev_bundle/deps/openssl-<version>. The main project only consumes the install (CMakeLists.txt: NODUS_OPENSSL_ROOT).
#
#   cmake -P scripts/build_openssl.cmake
#
# Linux needs perl and make (the bootstrap installs both). Windows needs, from a Visual Studio x64 developer prompt,
# a native perl (Strawberry Perl; not the MSYS/Git perl) and NASM on PATH.
#
# To move to a new release: change the version and the hash together. The hash is the one OpenSSL publishes
# (<tarball>.sha256 beside the release); check it against the release's signature once, by hand, when pinning it.
cmake_minimum_required(VERSION 3.20)

set(OPENSSL_PIN_VERSION "4.0.3")
set(OPENSSL_PIN_SHA256 "325b5c806167c13b40b1ffeadfe0248197c00eccc4cf123ec1e28d2d2fd216d9")
set(OPENSSL_PIN_URL
    "https://github.com/openssl/openssl/releases/download/openssl-${OPENSSL_PIN_VERSION}/openssl-${OPENSSL_PIN_VERSION}.tar.gz")

get_filename_component(_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(_deps "${_root}/dev_bundle/deps")
set(_prefix "${_deps}/openssl-${OPENSSL_PIN_VERSION}")
set(_work "${_deps}/_work/openssl-${OPENSSL_PIN_VERSION}")
set(_tarball "${_deps}/_downloads/openssl-${OPENSSL_PIN_VERSION}.tar.gz")

if(EXISTS "${_prefix}/.nodus-pin")
  file(READ "${_prefix}/.nodus-pin" _have)
  if(_have STREQUAL "${OPENSSL_PIN_SHA256}")
    message(STATUS "OpenSSL ${OPENSSL_PIN_VERSION} already installed at ${_prefix}")
    return()
  endif()
endif()

# The download is refused unless its bytes have the pinned hash; a tarball already here is checked the same way.
if(EXISTS "${_tarball}")
  file(SHA256 "${_tarball}" _got)
  if(NOT _got STREQUAL OPENSSL_PIN_SHA256)
    file(REMOVE "${_tarball}")
  endif()
endif()
if(NOT EXISTS "${_tarball}")
  message(STATUS "Downloading ${OPENSSL_PIN_URL}")
  file(DOWNLOAD "${OPENSSL_PIN_URL}" "${_tarball}" EXPECTED_HASH SHA256=${OPENSSL_PIN_SHA256} TLS_VERIFY ON STATUS _st)
  list(GET _st 0 _code)
  if(NOT _code EQUAL 0)
    file(REMOVE "${_tarball}")
    message(FATAL_ERROR "OpenSSL download failed: ${_st}")
  endif()
endif()

file(REMOVE_RECURSE "${_work}")
file(MAKE_DIRECTORY "${_work}")
file(ARCHIVE_EXTRACT INPUT "${_tarball}" DESTINATION "${_work}")
set(_src "${_work}/openssl-${OPENSSL_PIN_VERSION}")

# Static libraries (position-independent on ELF by OpenSSL's default), no tests, docs, command-line apps, legacy
# provider, engines, SSLv3, DTLS or QUIC. libcrypto keeps ML-KEM, ML-DSA, Ed25519, X25519, ChaCha20-Poly1305, AES-GCM
# and HKDF; libssl keeps TLS for the link's WebSocket.
set(_options no-shared no-tests no-docs no-apps no-legacy no-engine no-ssl3 no-dtls no-quic
    "--prefix=${_prefix}" "--openssldir=${_prefix}/ssl" "--libdir=lib")

find_program(_perl perl REQUIRED)
if(WIN32)
  find_program(_nasm nasm)
  if(NOT _nasm)
    message(FATAL_ERROR "NASM is not on PATH (winget install NASM.NASM, then reopen the developer prompt)")
  endif()
  execute_process(COMMAND "${_perl}" -e "print \$^O" OUTPUT_VARIABLE _perl_os)
  if(NOT _perl_os STREQUAL "MSWin32")
    message(FATAL_ERROR "${_perl} is not a native Windows perl (it reports '${_perl_os}'); use Strawberry Perl")
  endif()
  find_program(_nmake nmake REQUIRED)
  set(_configure "${_perl}" "${_src}/Configure" VC-WIN64A ${_options})
  set(_build "${_nmake}" build_libs)
  set(_install "${_nmake}" install_sw)
else()
  find_program(_make make REQUIRED)
  cmake_host_system_information(RESULT _jobs QUERY NUMBER_OF_LOGICAL_CORES)
  set(_configure "${_perl}" "${_src}/Configure" ${_options})
  set(_build "${_make}" -j${_jobs} build_libs)
  set(_install "${_make}" install_sw)
endif()

set(_bld "${_work}/build")
file(MAKE_DIRECTORY "${_bld}")
foreach(_step _configure _build _install)
  execute_process(COMMAND ${${_step}} WORKING_DIRECTORY "${_bld}" RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "OpenSSL ${_step} failed (${_rc})")
  endif()
endforeach()

file(WRITE "${_prefix}/.nodus-pin" "${OPENSSL_PIN_SHA256}")
file(REMOVE_RECURSE "${_deps}/_work")
message(STATUS "OpenSSL ${OPENSSL_PIN_VERSION} installed at ${_prefix}")
