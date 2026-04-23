$ErrorActionPreference = "Stop"

$body = @{
    question              = "批发要售圣女果是否免征增值税"
    policyId              = "KH1493204307733168128_20260422181920"
    version               = "v1"
    chunkSize             = 4000
    summaryBatchSize      = 3
    enableSkills          = $true
    enableRelations       = $true
    relationsExpansionMode = "all"
    summaryCleanAnswer    = $true
} | ConvertTo-Json -Compress

function Invoke-ReasonRun {
    param(
        [string]$Mode,
        [int]$RunIndex
    )

    $payload = ConvertFrom-Json $body
    $payload | Add-Member -NotePropertyName summaryPipelineMode -NotePropertyValue $Mode -Force
    $rawJson = ConvertTo-Json $payload -Compress -Depth 5
    $bodyBytes = [System.Text.Encoding]::UTF8.GetBytes($rawJson)

    $req = [System.Net.HttpWebRequest]::Create("http://127.0.0.1:5000/api/reason")
    $req.Method = "POST"
    $req.ContentType = "application/json; charset=utf-8"
    $req.ContentLength = $bodyBytes.Length
    $req.Timeout = 600000
    $req.ReadWriteTimeout = 600000

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $stream = $req.GetRequestStream()
    $stream.Write($bodyBytes, 0, $bodyBytes.Length)
    $stream.Close()

    try {
        $resp = $req.GetResponse()
    }
    catch [System.Net.WebException] {
        $sw.Stop()
        Write-Host "[$Mode][run$RunIndex] 请求失败: $($_.Exception.Message)" -ForegroundColor Red
        if ($_.Exception.Response) {
            $errStream = $_.Exception.Response.GetResponseStream()
            $reader = New-Object System.IO.StreamReader($errStream, [System.Text.Encoding]::UTF8)
            $errText = $reader.ReadToEnd()
            Write-Host $errText -ForegroundColor DarkRed
        }
        return $null
    }

    $respStream = $resp.GetResponseStream()
    $reader = New-Object System.IO.StreamReader($respStream, [System.Text.Encoding]::UTF8)
    $text = $reader.ReadToEnd()
    $reader.Close()
    $resp.Close()
    $sw.Stop()

    $outFile = Join-Path -Path "test" -ChildPath ("reason_pipeline_${Mode}_run${RunIndex}.json")
    [System.IO.File]::WriteAllText((Resolve-Path "test").Path + "\reason_pipeline_${Mode}_run${RunIndex}.json", $text, [System.Text.UTF8Encoding]::new($false))

    $obj = $text | ConvertFrom-Json
    $answer = $obj.data.answer
    $answerLen = if ($answer) { $answer.Length } else { 0 }
    Write-Host ("[{0}][run{1}] 耗时={2}s, status={3}, answerLen={4}" -f $Mode, $RunIndex, [math]::Round($sw.Elapsed.TotalSeconds, 2), $obj.status_code, $answerLen) -ForegroundColor Green

    return [pscustomobject]@{
        Mode      = $Mode
        Run       = $RunIndex
        Seconds   = [math]::Round($sw.Elapsed.TotalSeconds, 2)
        AnswerLen = $answerLen
        File      = "test/reason_pipeline_${Mode}_run${RunIndex}.json"
    }
}

$results = New-Object System.Collections.ArrayList

foreach ($mode in @("layered", "reduce_queue")) {
    for ($i = 1; $i -le 2; $i++) {
        $r = Invoke-ReasonRun -Mode $mode -RunIndex $i
        if ($r) { [void]$results.Add($r) }
    }
}

Write-Host ""
Write-Host "===== 汇总 =====" -ForegroundColor Cyan
$results | Format-Table -AutoSize
