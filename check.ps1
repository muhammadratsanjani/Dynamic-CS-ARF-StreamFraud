$bib = Get-Content references.bib -Raw
$tex = Get-Content main_manuscript.tex -Raw
$matches = [regex]::Matches($bib, '@[a-zA-Z]+\{([^,]+),')

$uncited = @()
foreach ($m in $matches) {
    $key = $m.Groups[1].Value
    if (-not ($tex -match $key)) {
        $uncited += $key
    }
}
$uncited | Out-File uncited.txt
