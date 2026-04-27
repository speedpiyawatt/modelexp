# Linux Portability Sweep v2

## Runtime Blockers

| Severity | File | Linux failure mode | Required fix | Retest command |
| --- | --- | --- | --- | --- |
| blocker | `wunderground/validate_history.py` | Default `--history-dir` pointed at a Mac-specific `/Users/...` path, so a Linux run without overrides failed immediately. | Switched the default to a repo-relative path derived from `__file__`. | `python3 wunderground/validate_history.py --help` |
| friction | `tools/nbm/build_grib2_features.py` | `wgrib2` discovery fell back to Mac-only install roots, so Linux users outside those layouts got misleading discovery behavior. | Kept lookup `PATH`-first and replaced Mac-only fallbacks with platform-neutral locations plus one explicit `WGRIB2_BINARY` override. | `python3 tools/nbm/build_grib2_features.py --help` |

## Verification / Setup Blockers

| Severity | File | Linux failure mode | Required fix | Retest command |
| --- | --- | --- | --- | --- |
| friction | `tools/weather/run_verification_suite.py` | Default output path assumed `/tmp`, and live-stage GRIB verification relied on ad hoc `wgrib2` lookup. | Switched the temp root to `tempfile.gettempdir()` and centralized `wgrib2` discovery using the same platform-neutral lookup rules. | `python3 tools/weather/run_verification_suite.py --help` |

## Documentation / Setup Drift

| Severity | File | Linux failure mode | Required fix | Retest command |
| --- | --- | --- | --- | --- |
| doc-only | `AGENTS.md` | Canonical validation command embedded an absolute Mac path. | Rewrote it to use the repo-relative history directory. | `rg -n "/Users/" AGENTS.md` |
| doc-only | `tools/nbm/README.md` | Intro text, example paths, and install guidance were Mac-centric and encouraged absolute local paths. | Rewrote commands and examples to be repo-relative and platform-neutral. | `rg -n "/Users/|/opt/homebrew|--user" tools/nbm/README.md` |
| doc-only | `tools/hrrr/README.md` | Setup examples assumed a Mac workspace path and `pip --user`. | Rewrote commands to run from repo-relative paths with neutral install wording. | `rg -n "/Users/|--user" tools/hrrr/README.md` |

## Sweep Result

- No runtime code defaults to `/Users/...`.
- No runtime tool discovery now depends on `/opt/homebrew` or other Mac-only locations.
- Core docs no longer require Mac-specific working directories or absolute paths for the reviewed commands.
