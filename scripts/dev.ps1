# ICMI development wrapper (Windows PowerShell).
#
# Same isolation contract as scripts/dev.sh: all ICMI caches and environments
# stay inside the repository directory; the machine-global uv cache used by
# other projects is never touched.
#
# Usage:  .\scripts\dev.ps1 <uv-args...>
#         .\scripts\dev.ps1 sync
#         .\scripts\dev.ps1 run pytest -q

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$env:UV_CACHE_DIR = Join-Path $RepoRoot ".uv-cache"
$env:UV_PYTHON_PREFERENCE = "only-system"

$EnvFile = Join-Path $RepoRoot ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | Where-Object { $_ -match '^\s*[^#<>]\s*[^=]+\s*=' } | ForEach-Object {
        $name, $value = $_ -split '=', 2
        Set-Item -Path "env:$($name.Trim())" -Value $value.Trim()
    }
}

& uv @args
exit $LASTEXITCODE
