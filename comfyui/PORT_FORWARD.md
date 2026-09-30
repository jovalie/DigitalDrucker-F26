# Port-forwarding ComfyUI from qclgpu

For anyone with **Cisco VPN** access (no Tailscale needed on your machine).
`<qcluser>` = your QCL username (e.g. `jzheng_cmc`).

## 1. Connect the Cisco (CMC) VPN

Both the hostname and the route to port 5022 depend on it.

## 2. Open the tunnel — keep this terminal open

```bash
ssh -N -L 8188:100.69.136.49:8188 -p 5022 <qcluser>@qclgpu.compute.cmc.edu
```

`-N` = no shell, just the forward. If your key isn't the default one, add `-i ~/.ssh/<yourkey>`.
If the hostname doesn't resolve, use the IP instead: `<qcluser>@134.173.177.241`.

## 3. Open the UI

**http://localhost:8188** — plain `http://`, **not** `https://`.

## The two gotchas

- **Target must be `100.69.136.49:8188`, not `localhost:8188`.** ComfyUI binds only to the
  machine's Tailscale IP, so a forward to loopback dies and the browser shows
  `NS_ERROR_NET_EMPTY_RESPONSE` (Firefox) / `ERR_EMPTY_RESPONSE` (Chrome). That empty response
  *is* the symptom of a wrong forward target, not a broken server.
- **`Could not resolve hostname qclgpu.compute.cmc.edu`** = VPN not up (or no DNS for the
  internal name). Either reconnect the VPN or pin it:
  `sudo sh -c 'echo "134.173.177.241 qclgpu.compute.cmc.edu" >> /etc/hosts'`

## Sanity checks

```bash
# through the tunnel — expect 200
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8188/system_stats

# on the host itself (does the service listen?)
ssh -p 5022 <qcluser>@qclgpu.compute.cmc.edu \
  'systemctl is-active comfyui; ss -ltn | grep 8188'
```

ComfyUI runs as the systemd unit `comfyui.service` (GPU0, port 8188). If it's not up, whoever has
sudo there can `sudo systemctl restart comfyui`; otherwise ask Joan. The workflows and models live
under `/mnt/raid/shared/comfyui/` (`models/`, `user/default/workflows/`).

## Persistent version (optional)

In `~/.ssh/config`:

```
Host qclgpu-fwd
    HostName qclgpu.compute.cmc.edu
    Port 5022
    User <qcluser>
    LocalForward 8188 100.69.136.49:8188
    ExitOnForwardFailure yes
    ServerAliveInterval 30
```

then `ssh -N qclgpu-fwd`.
