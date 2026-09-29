# Laguna access notes (CARC)

Sources: CMC QCL site (<https://cmc-qcl.github.io/HPC-research-computing/>) and the v1 plan.

**QCL's own GPU host (`qclgpu.compute.cmc.edu`) is a different system — see
[`../qcl/README.md`](../qcl/README.md).**

## Access path

1. **CARC account** — <https://hpcaccount.usc.edu> → choose school (SSO) → confirm assignment to
   the project **`CMC_Onboarding`**.
2. **SSH key registration** — either:
   - Portal: *User Profile → Public SSH Key* (propagation 1–2 h), or
   - **OnDemand (immediate):** <https://laguna-ood.carc.usc.edu> → Files → Home Directory →
     Show Dotfiles → `.ssh` → edit `authorized_keys` → append the public key → save.
3. **Login** (EPPN format): `ssh <EPPN>@laguna.carc.usc.edu`, e.g.
   `ssh JoaZheng@cmc.edu@laguna.carc.usc.edu`.
4. **Support:** `qcl@cmc.edu` (CMC QCL onboarding) · `laguna-support@usc.edu` (CARC).
   Allocation owner for this project: Yan Li (PI).

## Notes for this project

- Only the **Laguna regional cluster** is used; the **QCL DGX is excluded** (no VPN path).
- QCL's NSF ACCESS allocations (Bridges-2 GPU, Expanse GPU, Rockfish GPU, Anvil GPU, …) are
  overflow only and must be coordinated with QCL staff.
- Our pi-container public key (fingerprint `SHA256:J7US…`) is the one to register:
  ```
  ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINYkJ9lhi1PlIKb3cHJt6NXJneCvK5P+CHjnR0aRd/vW carc-pi
  ```
- If portal-registered keys are rejected, the OnDemand `authorized_keys` edit is the fastest
  workaround; if the account is not in `CMC_Onboarding`, email `qcl@cmc.edu`.
