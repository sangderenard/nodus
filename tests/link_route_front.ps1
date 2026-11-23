# link_route_front.ps1: the front route map through a real Apache, without touching a live front. It starts a private
# Apache on 127.0.0.1:<Port> whose only rule is the production link rule with the route map (RewriteMap txt:), runs
# link_route_test with that map and port (two links, each dialed by its own name through the front), and stops Apache.
#
#   .\tests\link_route_front.ps1 -Build .\build-win\Release [-Httpd C:\Apache24\bin\httpd.exe] [-Port 47690]
param(
    [Parameter(Mandatory = $true)][string]$Build,
    [string]$Httpd = 'C:\Apache24\bin\httpd.exe',
    [int]$Port = 47690
)
$ErrorActionPreference = 'Stop'
$work = Join-Path $env:TEMP ("link_route_front_" + [DateTimeOffset]::UtcNow.ToUnixTimeSeconds())
New-Item -ItemType Directory -Force $work | Out-Null
$root = Split-Path -Parent (Split-Path -Parent $Httpd)
$map = (Join-Path $work 'link.map') -replace '\\', '/'
Set-Content -Path $map -Value '# routes' -Encoding ascii
$conf = Join-Path $work 'front.conf'
# The [LINK] rule of the fronts (turing examples/meowdoku/deploy/apache), with the route map in front of the fixed
# upstream: a name in the map goes to its own loopback port; any other name to the default listener.
@"
ServerRoot "$($root -replace '\\', '/')"
Listen 127.0.0.1:$Port
ServerName 127.0.0.1
LoadModule authz_core_module modules/mod_authz_core.so
LoadModule rewrite_module modules/mod_rewrite.so
LoadModule proxy_module modules/mod_proxy.so
LoadModule proxy_http_module modules/mod_proxy_http.so
LoadModule proxy_wstunnel_module modules/mod_proxy_wstunnel.so
PidFile "$($work -replace '\\', '/')/httpd.pid"
ErrorLog "$($work -replace '\\', '/')/error.log"
LogLevel warn rewrite:trace2
Define LINK_UPSTREAM "127.0.0.1:47616"
Define LINK_ROUTES "$map"
RewriteEngine On
RewriteMap linkroutes "txt:`${LINK_ROUTES}"
RewriteCond %{HTTP:Upgrade} =websocket [NC]
RewriteRule ^/ws/link/([A-Za-z0-9._-]+)/(control|bulk)$ ws://`${linkroutes:`$1|`${LINK_UPSTREAM}}/ws/link/`$1/`$2 [P,L]
RewriteRule ^ - [R=404,L]
"@ | Set-Content -Path $conf -Encoding ascii

& $Httpd -t -f $conf
if ($LASTEXITCODE) { throw "the private front's configuration is refused" }
$apache = Start-Process $Httpd -ArgumentList @('-f', $conf) -PassThru -WindowStyle Hidden
try {
    $until = (Get-Date).AddSeconds(20)
    while ((Get-Date) -lt $until -and -not (Test-NetConnection 127.0.0.1 -Port $Port -InformationLevel Quiet -WarningAction SilentlyContinue)) { Start-Sleep -Milliseconds 300 }
    $env:Path = "$Build;$env:Path"
    & (Join-Path $Build 'link_route_test.exe') $work $map $Port 2>&1 |
        Where-Object { $_ -match '^(hosts|map|through|FAIL|FAILED|OK|#|mesh|someone)' }
    $code = $LASTEXITCODE
} finally {
    # Apache on Windows runs a parent and a child: end the tree by the parent's pid.
    & taskkill /PID $apache.Id /T /F 2>&1 | Out-Null
}
Write-Host "front log: $work\error.log"
exit $code
