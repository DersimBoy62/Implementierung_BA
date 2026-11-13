# Verifies hashes of every file inside the archive made by BA_ARCHIVIEREN_V54_DEVONLY.ps1.
# Does not execute any source file, load pickles or touch original model/data.
[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$ZipPath)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$zip = (Resolve-Path -LiteralPath $ZipPath).Path
$temp = Join-Path ([IO.Path]::GetTempPath()) ('v54_archive_verify_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temp | Out-Null
try {
    Expand-Archive -LiteralPath $zip -DestinationPath $temp -Force
    $mp = Join-Path $temp 'V54_ARCHIV_INTEGRITY_MANIFEST.json'
    if (-not (Test-Path -LiteralPath $mp -PathType Leaf)) { throw 'Manifest missing from archive' }
    $m = Get-Content -LiteralPath $mp -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($m.status -cne 'TECHNICAL_DEVELOPMENT_ARCHIVED_NOT_MODEL_FREEZE') { throw 'Invalid status in manifest' }
    if ($m.freeze_status -cne 'PROVISIONAL_USER_APPROVAL_PENDING') { throw 'Freeze status mismatch' }
    $n = 0
    foreach ($entry in $m.evidence_files) {
        $rel = [string]$entry.path
        if ($rel -match '(^|[\\/])\.\.([\\/]|$)' -or [IO.Path]::IsPathRooted($rel)) { throw 'Unsafe manifest path' }
        $p = Join-Path $temp $rel
        if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { throw ('Missing: ' + $rel) }
        $sha = (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($sha -ne [string]$entry.sha256) { throw ('Checksum failed: ' + $rel) }
        if ((Get-Item -LiteralPath $p).Length -ne [long]$entry.bytes) { throw ('Size mismatch: ' + $rel) }
        $n++
    }
    Write-Host ('PASS: ARCHIVE VERIFIED, ' + $n + ' files; development only, NOT formal model freeze')
    Write-Host ('ZIP SHA256: ' + (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant())
}
finally {
    if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp -Recurse -Force }
}
