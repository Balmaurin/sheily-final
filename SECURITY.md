# Security Policy

We take the security of Sheily AI seriously. This document explains how to report vulnerabilities and our security practices.

## Supported Versions

We currently support security patches on the `main` branch. Older tags may not receive updates.

## Reporting a Vulnerability

- Please email security reports to: security@sheily-ai.dev
- Do not open public issues for security reports.
- Include a detailed description, steps to reproduce, and potential impact.
- We aim to acknowledge reports within 72 hours and provide a status update within 7 days.

## Handling Sensitive Data

- Never commit secrets. This repository uses a `.secrets.baseline` and `detect-secrets` pre-commit hook.
- Use `.env` files locally (ignored by git) and CI secrets for pipelines.

## Hardening Measures

- Static analysis: Bandit runs in CI and pre-commit.
- Dependency scanning: Safety and pip-audit run in CI.
- Secret scanning: detect-secrets pre-commit with baseline.
- Least privilege: Docker runs as a non-root user.

## Related Docs

- See `docs/SECURITY_POLICIES.md` for broader security policies.
