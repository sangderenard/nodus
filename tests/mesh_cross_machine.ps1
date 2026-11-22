# mesh_cross_machine.ps1: the secured mesh between two machines over the production path. Windows is behind NAT, so it
# always dials out: wss://<front>/ws/link/mesh/{control,bulk}, TLS at Apache, which carries the socket to the host's one
# link listener (oracle-front.conf, 127.0.0.1:47616). Each node's keys are made on its own machine (nodus_mesh_keys);
# the writer and reader are mesh_secure_test's child modes, and MESH_TEST_LINK sets which end each one is.
#   1  W (Windows) writes, dialing; O (Linux) reads, listening: O opens W's canvas
#   2  W reads, dialing; O writes, listening: W opens O's canvas (who dials is independent of who writes)
#   3  W rotates; O accepts the record (made on Windows, checked on Linux); 1 again at W's epoch 1
# The host's listener port must be free for the run: -PauseCommand / -ResumeCommand stop and restore what holds it.
param(
    [Parameter(Mandatory)] [string]$LocalBuild,
    [Parameter(Mandatory)] [string]$Remote,
    [Parameter(Mandatory)] [string]$RemoteBuild,
    [string]$Front = 'oracle.speaktome.dev',
    [int]$Port = 47616,
    [string]$SshKey = "$HOME\.ssh\oracle_nodus",
    [string]$RemoteUser = 'nodus',
    [string]$PauseCommand = '',
    [string]$ResumeCommand = '',
    [int]$Seconds = 25
)
$ErrorActionPreference = 'Stop'
$stamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$work = Join-Path $env:TEMP "mesh_cross_$stamp"
$rwork = "/tmp/mesh_cross_$stamp"
New-Item -ItemType Directory -Force $work | Out-Null
$env:Path = "$LocalBuild;$env:Path"
$test = Join-Path $LocalBuild 'mesh_secure_test.exe'
$keys = Join-Path $LocalBuild 'nodus_mesh_keys.exe'
$ssh = @('-i', $SshKey, '-o', 'BatchMode=yes')
$rows = 'canvas_row', 'tensor_mint', 'tensor_returns'

function Invoke-Remote([string]$command) {
    $out = & ssh @ssh $Remote "sudo -u $RemoteUser bash -c '$command'" 2>&1
    if ($LASTEXITCODE -ne 0) { throw "remote: $command`n$out" }
    return ($out -join "`n")
}
function Get-Kid([string]$report) { if ($report -match 'kid ([0-9a-f]{64})') { return $Matches[1] }; throw "no kid in: $report" }
function Wait-Line([string]$file, [string]$text, [int]$seconds) {
    $until = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $until) {
        if ((Test-Path $file) -and (Select-String -Path $file -SimpleMatch $text -Quiet)) { return $true }
        Start-Sleep -Milliseconds 300
    }
    return $false
}

# One pair. The Linux node listens on $Port (MESH_TEST_LINK=host:<key>); the Windows node dials the front.
function Invoke-Pair([string]$name, [bool]$windowsWrites, [string]$writerKid, [string]$readerKid) {
    $t = (& $test tickets) -split ' '
    $linkKey, $ct, $bt = $t[0], $t[1], $t[2]
    $readerOut = Join-Path $work "$name.reader.txt"
    $env:NODUS_MESH_KEYS = $W
    $env:MESH_TEST_LINK = "wss:${Front}:${ct}:${bt}"
    $hostEnv = "NODUS_MESH_KEYS=$rwork/O MESH_TEST_LINK=host:$linkKey"
    $local, $remote = $null, $null
    # The cleanup is in a finally: a node that never comes up throws, and the remote test must not keep the listener
    # port until its own 420/600-second timeout.
    try {
        if ($windowsWrites) {
            $remote = Start-Process ssh -ArgumentList ($ssh + @($Remote, "sudo -u $RemoteUser bash -c 'cd $rwork && $hostEnv timeout 600 $RemoteBuild/mesh_secure_test reader $Port $linkKey $writerKid $Seconds $($rows -join ' ') mesh_verify mesh_open'")) `
                      -PassThru -WindowStyle Hidden -RedirectStandardOutput $readerOut -RedirectStandardError "$readerOut.err"
            if (-not (Wait-Line $readerOut '[reader] up' 300)) { throw "${name}: the Linux reader did not come up" }
            $local = Start-Process $test -ArgumentList (@('writer', $work, $Port, $ct, $bt, $readerKid) + $rows + @('mesh_seal', 'mesh_sign')) `
                     -WorkingDirectory $work -PassThru -WindowStyle Hidden -RedirectStandardOutput "$work\$name.writer.txt" -RedirectStandardError "$work\$name.writer.err"
            $remote | Wait-Process -Timeout 600 -ErrorAction SilentlyContinue
        } else {
            $remote = Start-Process ssh -ArgumentList ($ssh + @($Remote, "sudo -u $RemoteUser bash -c 'cd $rwork && $hostEnv timeout 420 $RemoteBuild/mesh_secure_test writer $rwork $Port $ct $bt $readerKid $($rows -join ' ') mesh_seal mesh_sign'")) `
                      -PassThru -WindowStyle Hidden -RedirectStandardOutput "$work\$name.writer.txt" -RedirectStandardError "$work\$name.writer.err"
            if (-not (Wait-Line "$work\$name.writer.txt" '[writer] up' 300)) { throw "${name}: the Linux writer did not come up" }
            $local = Start-Process $test -ArgumentList (@('reader', $Port, $linkKey, $writerKid, $Seconds) + $rows + @('mesh_verify', 'mesh_open')) `
                     -WorkingDirectory $work -PassThru -WindowStyle Hidden -RedirectStandardOutput $readerOut -RedirectStandardError "$readerOut.err"
            $local | Wait-Process -Timeout 600 -ErrorAction SilentlyContinue
        }
    } finally {
        foreach ($p in $local, $remote) { if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } }
        try { Invoke-Remote "pkill -f mesh_secure_test.*$Port || true" | Out-Null } catch { Write-Host "  remote cleanup: $_" }
        Remove-Item Env:MESH_TEST_LINK -ErrorAction SilentlyContinue
    }
    foreach ($f in "$work\$name.writer.txt", $readerOut) {    # each change of link state, with the endpoint's reason
        if (Test-Path $f) { Select-String -Path $f -Pattern '^\[(writer|reader)\] link' | ForEach-Object { Write-Host "  $($_.Line)" } }
    }
    $result = Select-String -Path $readerOut -Pattern '^RESULT' | Select-Object -Last 1
    $line = if ($result) { $result.Line } else { 'RESULT none' }
    Write-Host "$name : $line"
    $readable = if ($line -match 'readable_generations (\d+)') { [int]$Matches[1] } else { 0 }
    return [pscustomobject]@{ Name = $name; Linked = ($line -match 'linked 1 '); Readable = $readable }
}

# Keys, each made on its own machine; public blocks exchanged (key files are 0600, so copied out by their owner).
$W = Join-Path $work 'W'
$kidW = Get-Kid (& $keys keygen $W)
$kidO = Get-Kid (Invoke-Remote "mkdir -p $rwork && $RemoteBuild/nodus_mesh_keys keygen $rwork/O")
Invoke-Remote "cp $rwork/O/epochs/0/node.pub /tmp/O_$stamp.pub && chmod 644 /tmp/O_$stamp.pub" | Out-Null
& scp @('-i', $SshKey, '-q') "${Remote}:/tmp/O_$stamp.pub" (Join-Path $W "peers\$kidO.pub")
& scp @('-i', $SshKey, '-q') (Join-Path $W 'epochs\0\node.pub') "${Remote}:/tmp/W_$stamp.pub"
Invoke-Remote "cp /tmp/W_$stamp.pub $rwork/O/peers/$kidW.pub" | Out-Null
Write-Host "keys: W $($kidW.Substring(0,8)).. (Windows)  O $($kidO.Substring(0,8)).. (Linux); front $Front, listener $Port"

$results = @()
try {
    if ($PauseCommand) { Write-Host 'pausing the listener port holder'; & ssh @ssh $Remote $PauseCommand }
    $results += Invoke-Pair 'W writes (dials), O reads (listens)' $true $kidW $kidO
    $results += Invoke-Pair 'W reads (dials), O writes (listens)' $false $kidO $kidW
    Write-Host (& $keys rotate $W 600)
    & scp @('-i', $SshKey, '-q') (Join-Path $W 'rotation-1.rec') "${Remote}:/tmp/W_rot_$stamp.rec"
    Write-Host ('O: ' + ((Invoke-Remote "cp /tmp/W_rot_$stamp.rec $rwork/W-rotation-1.rec && $RemoteBuild/nodus_mesh_keys accept $rwork/O $rwork/W-rotation-1.rec") -split "`n" | Select-Object -Last 1))
    $results += Invoke-Pair 'W at epoch 1 writes (dials), O reads' $true $kidW $kidO
} finally {
    if ($ResumeCommand) { Write-Host 'restoring the listener port holder'; & ssh @ssh $Remote $ResumeCommand }
}
$failed = @($results | Where-Object { -not $_.Linked -or $_.Readable -lt 1 })
if ($failed.Count -or $results.Count -lt 3) { Write-Host "FAILED: $($failed.Name -join '; ')"; exit 1 }
Write-Host 'OK'
