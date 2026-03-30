# SwordBrawl

Multiplayer sword fighting powered by physics simulation (PhysX WASM) and neural network character control (ONNX Runtime WASM), running on a PartyKit server.

## Play

Visit the live demo: https://zalo.github.io/SwordBrawl/

## Architecture

- **Server (PartyKit Durable Object)**: Runs PhysX WASM for physics simulation and ONNX Runtime WASM for neural network inference. Each player gets a full humanoid articulation. The server steps the simulation, runs the HRL policy (HLC + LLC) per player, and broadcasts link poses to all clients.
- **Client (GitHub Pages)**: Three.js rendering only. Sends player inputs (WASD, actions) over WebSocket, receives humanoid poses, and renders all players with floating name labels.

## Controls

- **WASD** — Move
- **Right-click drag** — Rotate camera
- **Scroll** — Zoom
- **Left click** — Slash
- **Space** — Jump
- **Q** — Kick
- **E** — Block

Touch controls (joystick + action buttons) are available on mobile.

## Development

```sh
npm install
npm run dev
```

## Deploy

```sh
npm run deploy
```

## Tech

- [PartyKit](https://partykit.io) for multiplayer server infrastructure
- [PhysX WASM](https://github.com/nicholasgasior/physx-js-webidl) for rigid body / articulation simulation
- [ONNX Runtime Web](https://onnxruntime.ai/) (custom WASM integration for Cloudflare Workers)
- [Three.js](https://threejs.org) for 3D rendering
- Hierarchical Reinforcement Learning (HRL) for character control
