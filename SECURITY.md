# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities through GitHub private vulnerability reporting when available. Do not include real API credentials, private media, personal paths, generated identity images, transcripts or provider responses in a public issue.

## Credential handling

This project does not require credentials to be stored in the repository. Cloud-provider credentials are accepted at runtime through a user-selected credential file, a node widget or a supported process variable. The supplied workflows keep every credential field empty.

## Release checks

`tools/audit_release.py` rejects common API-key formats, private-key headers, absolute personal paths and media files. GitHub Actions runs the same check for every push and pull request.
