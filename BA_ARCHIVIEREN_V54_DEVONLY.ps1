# Bayesian Bachelor — V5.4 TECHNICAL DEVELOPMENT SNAPSHOT (NOT A MODEL FREEZE)
# Run from Implementierung_BA; no training, unpickling, predictions or quality evaluations.
# Source files are never modified. Archive created once; if output exists, script fails.
[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path,
    [string]$ArchiveName = 'V54_ENTWICKLUNGSSTAND_ARCHIV_2026-11-12'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$ExpectedDataSHA = 'bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5'
$ExpectedConfigSHA = '88f9da6aa51be71db7968a8222cb36e5e4838b8a8431988bd7c54c09bace51ab'
$ExpectedModelSHA = '56c3605ba3e0157125ecb7b2adaaddcb160a15c16edf22f2e28e5b232f2ae3c4'
$ExpectedL = @('DM04','DM05','DM06','DM09','DM12','DM16','DM17','DM18')
$ExpectedU = @('DM01','DM02','DM10','DM14')
$ExpectedLocked = @('DM07','DM08','DM11','DM15')
function Assert-True([bool]$ok, [string]$msg) { if (-not $ok) { throw $msg } }
function FileSHA([string]$p) { return (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant() }
function Equal-Ids($actual, $expected) { return (($actual -join ',') -ceq ($expected -join ',')) }
# Windows / OneDrive: a genuine Cloud Files placeholder also has the ReparsePoint
# attribute. Do NOT blindly deny that bit: inspect its native reparse tag instead.
# Permit only Windows Cloud Files API tags 0x9000001A .. 0x9000F01A;
# symbolic links (0xA000000C), junctions (0xA0000003), and all unknown tags fail closed.
Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

public static class V54ArchiveNativePathCheck
{
    [StructLayout(LayoutKind.Sequential)]
    private struct FILE_ATTRIBUTE_TAG_INFO
    {
        public uint FileAttributes;
        public uint ReparseTag;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true, EntryPoint = "CreateFileW")]
    private static extern SafeFileHandle CreateFile(
        string name, uint desiredAccess, uint shareMode,
        IntPtr securityAttributes, uint creationDisposition,
        uint flagsAndAttributes, IntPtr templateFile);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetFileInformationByHandleEx(
        SafeFileHandle handle, int infoClass,
        out FILE_ATTRIBUTE_TAG_INFO info, uint bufferSize);

    public static uint GetReparseTag(string path)
    {
        // Open the file itself, not the target of its final reparse point.
        const uint FILE_READ_ATTRIBUTES = 0x00000080u;
        const uint FILE_SHARE_ALL = 0x00000007u;
        const uint OPEN_EXISTING = 3u;
        const uint FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000u;
        using (SafeFileHandle handle = CreateFile(path, FILE_READ_ATTRIBUTES,
            FILE_SHARE_ALL, IntPtr.Zero, OPEN_EXISTING,
            FILE_FLAG_OPEN_REPARSE_POINT, IntPtr.Zero))
        {
            if (handle.IsInvalid)
                throw new Win32Exception(Marshal.GetLastWin32Error(),
                    "Cannot inspect reparse tag: " + path);
            FILE_ATTRIBUTE_TAG_INFO info;
            if (!GetFileInformationByHandleEx(handle, 9, out info,
                (uint)Marshal.SizeOf(typeof(FILE_ATTRIBUTE_TAG_INFO))))
                throw new Win32Exception(Marshal.GetLastWin32Error(),
                    "Cannot read FileAttributeTagInfo: " + path);
            if ((info.FileAttributes & 0x00000400u) == 0)
                return 0u; // Not a reparse point.
            return info.ReparseTag;
        }
    }

    public static bool IsCloudFilesTag(uint tag)
    {
        return (tag & 0xFFFF0FFFu) == 0x9000001Au;
    }
}
'@
function Assert-SafeArchiveInput([string]$p) {
    $tag = [V54ArchiveNativePathCheck]::GetReparseTag($p)
    if ($tag -ne 0) {
        Assert-True ([V54ArchiveNativePathCheck]::IsCloudFilesTag($tag)) ('Unsafe symbolic link, junction or unrecognized reparse tag 0x' + $tag.ToString('X8') + ': ' + $p)
        Write-Host ('INFO: Verified OneDrive/Cloud Files placeholder (tag 0x' + $tag.ToString('X8') + '): ' + $p)
    }
}

$root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$folder = Join-Path $root 'results'
Assert-True (Test-Path -LiteralPath $folder -PathType Container) 'Missing results directory'
Assert-True ($ArchiveName -match '^V54_[A-Za-z0-9_\-]+$') 'ArchiveName must start with V54_ and use safe characters'
$destination = Join-Path $folder ($ArchiveName + '.zip')
Assert-True (-not (Test-Path -LiteralPath $destination)) 'Archive already exists: do not overwrite. Choose a new ArchiveName.'
$configPath = Join-Path $root 'final_candidate_v52\PROPOSED_MODEL_FREEZE.json'
$modelPath = Join-Path $root 'results\v54_development_replay\MODEL_8L4U_RUN01\TRAINED_MODEL.pkl'
$modelManifestPath = Join-Path $root 'results\v54_development_replay\MODEL_8L4U_RUN01\TRAIN_MANIFEST.json'
$dataPath = Join-Path $root 'input\DM_16_aktive_Ringe.pkl'
foreach ($p in @($configPath,$modelPath,$modelManifestPath,$dataPath)) {
    Assert-True (Test-Path -LiteralPath $p -PathType Leaf) ('Missing file: ' + $p)
    Assert-SafeArchiveInput $p
}
Assert-True ((FileSHA $configPath) -eq $ExpectedConfigSHA) 'Registered config SHA256 mismatch'
Assert-True ((FileSHA $modelPath) -eq $ExpectedModelSHA) 'Trained model SHA256 mismatch'
Assert-True ((FileSHA $dataPath) -eq $ExpectedDataSHA) 'Source dataset SHA256 mismatch (dataset will not be copied)'
$config = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
Assert-True ($config.freeze_status -ceq 'PROVISIONAL_USER_APPROVAL_PENDING') 'Model configuration was modified or prematurely approved'
Assert-True (Equal-Ids $config.labeled_ring_ids $ExpectedL) 'Wrong L ring IDs'
Assert-True (Equal-Ids $config.unlabeled_ring_ids $ExpectedU) 'Wrong U ring IDs'
Assert-True (Equal-Ids $config.never_use_for_selection_ids $ExpectedLocked) 'Wrong locked historic test IDs'
$core = Join-Path $root 'final_candidate_v52\final_core'
$coreItems = @($config.core_math_file_sha256.PSObject.Properties)
Assert-True ($coreItems.Count -eq 10) 'Expected exactly ten registered mathematical core hashes'
foreach ($f in $coreItems) {
    $path = Join-Path $core $f.Name
    Assert-True (Test-Path -LiteralPath $path -PathType Leaf) ('Missing core module: ' + $f.Name)
    Assert-True ((FileSHA $path) -eq [string]$f.Value) ('Core module altered: ' + $f.Name)
}
$trainMan = Get-Content -LiteralPath $modelManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
Assert-True ($trainMan.model_sha256 -eq $ExpectedModelSHA) 'Train manifest model SHA mismatch'
Assert-True ($trainMan.config_sha256 -eq $ExpectedConfigSHA) 'Train manifest config SHA mismatch'
Assert-True (Equal-Ids $trainMan.train_labeled_ids $ExpectedL) 'Train manifest L mismatch'
Assert-True (Equal-Ids $trainMan.train_U_process_only_ids $ExpectedU) 'Train manifest U mismatch'
Assert-True (Equal-Ids $trainMan.locked_old_test_ids_not_used $ExpectedLocked) 'Train manifest old-test lock mismatch'
# Strict published-code allowlist: never archive user-created extra files in QA directories.
$publishedFiles = @(
  'final_candidate_v52\CLOSEOUT_TEMPLATE.json',
  'final_candidate_v52\DEVELOPMENT_INPUT_POLICY_V51.json',
  'final_candidate_v52\PROPOSED_MODEL_FREEZE.json',
  'final_candidate_v52\PRUEFBERICHT_V51_EINGANGSSICHERHEIT_KAPITEL4_5_2026-11-12.docx',
  'final_candidate_v52\PRUEFBERICHT_V52_KONFIGURATION_PERSISTENZ_REPRODUZIERBARKEIT_2026-11-12.docx',
  'final_candidate_v52\README_V51_INPUT_SAFETY.md',
  'final_candidate_v52\README_V52_CONFIG_PERSISTENZ.md',
  'final_candidate_v52\RELEASE_SHA256_MANIFEST.json',
  'final_candidate_v52\V51_QA_RESULTS.json',
  'final_candidate_v52\V52_QA_RESULTS.json',
  'final_candidate_v52\V52_RELEASE_MANIFEST.json',
  'final_candidate_v52\final_core\compare_dynamic_quality_cv4.py',
  'final_candidate_v52\final_core\compare_two_head_cv4.py',
  'final_candidate_v52\final_core\dm_data.py',
  'final_candidate_v52\final_core\dynamic_markov_v2.py',
  'final_candidate_v52\final_core\dynamic_quality_v2.py',
  'final_candidate_v52\final_core\full_markov_b1.py',
  'final_candidate_v52\final_core\fullcov_b1.py',
  'final_candidate_v52\final_core\quality_link_challenge.py',
  'final_candidate_v52\final_core\two_head_b1.py',
  'final_candidate_v52\final_core\vb_b1.py',
  'final_candidate_v52\input_safety_v51.py',
  'final_candidate_v52\main_bayesian_ssr_v51.py',
  'final_candidate_v52\main_bayesian_ssr_v52.py',
  'final_candidate_v52\release_integrity_v52.py',
  'final_candidate_v52\test_development_logger_equivalence_v51.py',
  'final_candidate_v52\test_input_safety_v51.py',
  'final_candidate_v52\test_release_integrity_v52.py',
  'qa_v53\PRUEFBERICHT_V53_NUMERISCHE_VORPRUEFUNG_2026-11-12.docx',
  'qa_v53\README_V53_NUMERICAL_QA.md',
  'qa_v53\V53_QA_REFERENCE_LINUX.json',
  'qa_v53\V53_SHA256_MANIFEST.json',
  'qa_v53\test_numerical_v53.py',
  'qa_v54_elbo_audit\QA_SHA256_MANIFEST.json',
  'qa_v54_elbo_audit\README.md',
  'qa_v54_elbo_audit\audit_model_history_v54.py',
  'qa_v54_freeze_gate\README_V54_PRE_FREEZE_QA.md',
  'qa_v54_freeze_gate\V54_LINUX_QA_REFERENCE.json',
  'qa_v54_freeze_gate\V54_QA_SHA256_MANIFEST.json',
  'qa_v54_freeze_gate\pre_freeze_gate_v54.py',
  'qa_v54_prediction_boundary\README.md',
  'qa_v54_prediction_boundary\SHA256_MANIFEST.json',
  'qa_v54_prediction_boundary\test_prediction_boundary_v54.py',
  'qa_v54_real_replay\README_V54_REAL_DEV_REPLAY.md',
  'qa_v54_real_replay\V54_REAL_REPLAY_SHA256_MANIFEST.json',
  'qa_v54_real_replay\check_v52_development_replay.py',
  'research_v53_feedback\README_V53_FEEDBACK_DEVONLY.md',
  'research_v53_feedback\V53_FEEDBACK_SHA256_MANIFEST.json',
  'research_v53_feedback\paired_feedback_dev_v53.py',
  'research_v53_feedback\test_paired_feedback_smoke_v53.py'
)
$namedFiles = @(
  'results\v53_feedback_devonly\FOLD_1_PAIRED_STEPS_0_8.json',
  'results\v53_feedback_devonly\FOLD_2_PAIRED_STEPS_0_8.json',
  'results\v53_feedback_devonly\FOLD_3_PAIRED_STEPS_0_8.json',
  'results\v53_feedback_devonly\FOLD_4_PAIRED_STEPS_0_8.json',
  'results\v53_feedback_devonly\V53_PAIRED_DEVELOPMENT_CV4_SUMMARY.json',
  'results\v54_pre_freeze\V54_ORIGINAL_CV4_AUDIT.json',
  'results\v54_development_replay\MODEL_8L4U_RUN01\TRAINED_MODEL.pkl',
  'results\v54_development_replay\MODEL_8L4U_RUN01\TRAIN_MANIFEST.json',
  'results\v54_development_replay\V54_REAL_DEV_REPLAY_AUDIT_01.json',
  'results\v54_development_replay\V54_ELBO_HISTORY_AUDIT_01.json',
  'results\v54_development_replay\V54_PREDICTION_BOUNDARY_AUDIT_01.json',
  'BA_ARCHIVIEREN_V54_DEVONLY.ps1',
  'BA_ARCHIV_PRUEFEN_V54.ps1',
  'DOKUMENTATION_V54\BA_Kapitel_4_5_Wissenschaftliche_Ergebnis_Diskussionsbausteine_V54_2026-11-12.docx',
  'DOKUMENTATION_V54\BA_V54_Archivierung_und_Formaler_Freeze_Entscheidungsvorlage_2026-11-12.docx',
  'DOKUMENTATION_V54\FREEZE_ENTSCHEIDUNG_V54_NOCH_OFFEN.json',
  'DOKUMENTATION_V54\README_V54_FORMAL_FREEZE_VORBEREITUNG.md'
)
$stage = Join-Path $folder ('.v54_archive_stage_' + [guid]::NewGuid().ToString('N'))
$tempZip = Join-Path $folder ('.v54_archive_' + [guid]::NewGuid().ToString('N') + '.zip')
Assert-True (-not (Test-Path -LiteralPath $stage)) 'Stage collision'
New-Item -ItemType Directory -Path $stage | Out-Null
try {
    $copied = New-Object 'System.Collections.Generic.List[string]'
    foreach ($rel in $publishedFiles) {
        $src = Join-Path $root $rel
        Assert-True (Test-Path -LiteralPath $src -PathType Leaf) ('Missing published file: ' + $rel)
        Assert-SafeArchiveInput $src
        $dest = Join-Path $stage $rel
        $parent = Split-Path -Path $dest -Parent
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
        Copy-Item -LiteralPath $src -Destination $dest
        Assert-True ((FileSHA $src) -eq (FileSHA $dest)) ('Published file copy failed: ' + $rel)
        $copied.Add($rel.Replace('\','/'))
    }
    foreach ($rel in $namedFiles) {
        $src = Join-Path $root $rel
        Assert-True (Test-Path -LiteralPath $src -PathType Leaf) ('Missing evidence file: ' + $rel)
        Assert-SafeArchiveInput $src
        $dest = Join-Path $stage $rel
        $parent = Split-Path -Path $dest -Parent
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
        Copy-Item -LiteralPath $src -Destination $dest
        Assert-True ((FileSHA $src) -eq (FileSHA $dest)) ('Evidence copy failed: ' + $rel)
        $copied.Add($rel.Replace('\','/'))
    }
    $fileRows = @($copied | Sort-Object -Unique | ForEach-Object {
        $entry = $_
        $p = Join-Path $stage $entry
        [ordered]@{ path = $entry; sha256 = (FileSHA $p); bytes = (Get-Item -LiteralPath $p).Length }
    })
    $audit = [ordered]@{
      schema = 'V54_DEVONLY_ARCHIVE_2026-11-12_v1'
      status = 'TECHNICAL_DEVELOPMENT_ARCHIVED_NOT_MODEL_FREEZE'
      archive_created_utc = (Get-Date).ToUniversalTime().ToString('o')
      scientific_status = 'Exploratory repeated 8-development-label CV; no independent generalization evidence'
      data_file_sha256_verified_but_not_included = $ExpectedDataSHA
      config_sha256 = $ExpectedConfigSHA
      model_sha256 = $ExpectedModelSHA
      prediction_fingerprint_as_reported = '9c0566b93663297570c9a4a2ff5f36abc5196e1f81bb910d182620756c6dfc52'
      freeze_status = 'PROVISIONAL_USER_APPROVAL_PENDING'
      training_L = $ExpectedL
      training_U_masked = $ExpectedU
      locked_historical_test = $ExpectedLocked
      evidence_files = $fileRows
    }
    $auditFile = Join-Path $stage 'V54_ARCHIV_INTEGRITY_MANIFEST.json'
    ($audit | ConvertTo-Json -Depth 20) | Set-Content -LiteralPath $auditFile -Encoding UTF8
    Assert-True ($fileRows.Count -ge 20) 'Unexpectedly small archive file count'
    Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $tempZip -CompressionLevel Optimal
    Assert-True (Test-Path -LiteralPath $tempZip -PathType Leaf) 'ZIP creation failed'
    Assert-True (-not (Test-Path -LiteralPath $destination)) 'Archive destination unexpectedly exists; refusing overwrite'
    Move-Item -LiteralPath $tempZip -Destination $destination
    Write-Host 'PASS: V5.4 DEVELOPMENT ARCHIVE WRITTEN ONCE; NOT A FORMAL MODEL FREEZE'
    Write-Host ('ZIP: ' + $destination)
    Write-Host ('ZIP SHA256: ' + (FileSHA $destination))
    Write-Host ('Archived files: ' + $fileRows.Count)
    Write-Host 'Source DM Pickle: hash checked, NOT included or deserialized'
}
finally {
    if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
    if (Test-Path -LiteralPath $tempZip) { Remove-Item -LiteralPath $tempZip -Force }
}
