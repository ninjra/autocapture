$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path "$PSScriptRoot/.."
$ResultsDir = Join-Path $RootDir "bench/results"
$TraceDir = Join-Path $ResultsDir "traces"
$CaseId = if ($env:BENCH_CASE_ID) { $env:BENCH_CASE_ID } else { "case1" }
$Iterations = if ($env:BENCH_ITERATIONS) { [int]$env:BENCH_ITERATIONS } else { 20 }
$Warmup = if ($env:BENCH_WARMUP) { [int]$env:BENCH_WARMUP } else { 3 }
$Trace = if ($env:BENCH_TRACE) { $env:BENCH_TRACE } else { "0" }
$ReplayDir = Join-Path $RootDir "bench/fixtures/responses"
$TimingFile = Join-Path $ResultsDir "timing.tsv"
$SummaryFile = Join-Path $ResultsDir "summary.json"
$OutputFile = Join-Path $ResultsDir "output_$CaseId.txt"

New-Item -ItemType Directory -Force -Path $ResultsDir, $TraceDir | Out-Null
"iteration`tms" | Set-Content -Encoding utf8 $TimingFile

function Run-Once([string]$TraceFile) {
  if ($TraceFile) {
    poetry run python -m autocapture.bench.llm_bench `
      --case-id $CaseId `
      --offline `
      --replay-dir $ReplayDir `
      --output $OutputFile `
      --trace-timing `
      --trace-timing-file $TraceFile
  } else {
    poetry run python -m autocapture.bench.llm_bench `
      --case-id $CaseId `
      --offline `
      --replay-dir $ReplayDir `
      --output $OutputFile
  }
}

for ($i = 1; $i -le $Warmup; $i++) {
  Run-Once ""
}

for ($i = 1; $i -le $Iterations; $i++) {
  $TraceFile = ""
  if ($Trace -eq "1") {
    $TraceFile = Join-Path $TraceDir "run_$i.jsonl"
  }
  $elapsed = (Measure-Command { Run-Once $TraceFile }).TotalMilliseconds
  $line = "$i`t$([math]::Round($elapsed, 0))"
  Add-Content -Encoding utf8 $TimingFile $line
}

poetry run python -m autocapture.bench.summary `
  --input $TimingFile `
  --output $SummaryFile `
  --case-id $CaseId `
  --mode offline
