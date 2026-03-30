// @ts-check
/** @typedef {import("partykit/server").Room} Room */
/** @typedef {import("partykit/server").Server} Server */
/** @typedef {import("partykit/server").Connection} Connection */
/** @typedef {import("partykit/server").ConnectionContext} ConnectionContext */

import PhysXInit from '../assets/physx/physx-js-webidl.mjs';
import physxWasm from '../assets/physx/physx-js-webidl.wasm';
import { initOrt, createSession, runSession } from './ort-server.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const DT = 1 / 120;
const CONTROL_DT = 1 / 30;
const NUM_SUBSTEPS = 4;
const TICK_HZ = 30;
const STATE_HZ = 20;        // broadcast pose state at 20 Hz
const RESET_GRACE = 90;
const MAX_PLAYERS = 8;

const PLAYER_COLORS = [
  '#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22','#e84393'
];

// Action latent vectors (hardcoded from ASE training)
const ACTION_LATENTS = {
  slash: new Float32Array([0.0681,-0.1512,0.0828,0.0363,0.1679,0.192,-0.1132,0.1329,0.1523,0.0772,-0.0797,0.0543,0.1023,-0.1551,0.03,-0.0118,-0.0298,0.1185,0.1505,-0.0285,-0.1074,-0.1345,0.1648,-0.0676,0.0423,0.1417,-0.0001,0.0669,0.0213,-0.1006,0.1132,-0.051,-0.1751,0.1089,0.1207,0.11,-0.063,-0.0133,0.035,0.0757,0.216,-0.191,-0.0713,-0.1795,-0.0725,0.1694,-0.1799,-0.2689,-0.1773,-0.0588,-0.0261,0.0526,0.0982,-0.092,0.1461,-0.0359,-0.07,-0.2553,0.2099,-0.231,0.0232,-0.0462,0.2094,0.1014]),
  kick:  new Float32Array([-0.1726,0.2087,0.0136,-0.1687,0.029,0.0078,0.0618,-0.1522,0.0816,-0.0158,-0.066,0.0249,0.008,-0.109,-0.115,-0.1335,0.1927,0.2766,-0.2196,-0.1182,0.049,0.0185,-0.0841,-0.0124,-0.0779,-0.1243,-0.0456,0.0009,0.3156,0.1657,0.1577,-0.088,-0.3035,-0.1513,-0.0079,0.0822,-0.0621,-0.0472,0.1181,0.0682,0.157,-0.1083,-0.1816,-0.0351,-0.1332,-0.1885,-0.0418,-0.1103,0.0936,0.1706,-0.0301,0.0358,0.1496,0.0992,-0.0414,0.1192,-0.0524,0.0195,-0.0299,-0.0236,0.1757,-0.1478,0.1142,0.0575]),
  block: new Float32Array([0.0147,-0.1071,-0.0575,-0.0925,0.0194,-0.2383,0.0202,-0.0408,-0.1094,0.1627,-0.0201,-0.0478,-0.0014,0.1708,0.0136,-0.1677,-0.1777,0.2129,0.0415,0.0839,0.0239,-0.0046,0.3436,-0.1045,0.0356,0.0851,-0.0842,0.1067,0.2573,0.1408,-0.105,-0.0626,-0.0018,-0.0191,0.2365,-0.1052,0.0477,0.0649,0.0286,0.183,-0.0381,-0.0238,0.1371,0.2019,0.0614,0.1226,0.1686,0.0353,0.1189,0.1487,0.1027,-0.0746,0.175,-0.2154,-0.0175,0.0893,0.0313,-0.0709,0.2553,-0.0238,-0.2038,0.0321,0.0597,-0.0405]),
  jump:  new Float32Array([-0.0278,-0.1597,0.0535,-0.0853,-0.063,-0.0871,-0.0606,0.2214,-0.1453,0.0232,0.2877,0.1577,0.2037,0.1257,0.0539,-0.1776,-0.0447,0.0179,-0.0674,-0.0797,0.0667,-0.0078,0.0265,-0.177,0.1485,0.0266,-0.1997,-0.2082,0.0736,-0.1618,-0.1899,-0.2072,0.0501,0.1432,0.0351,0.1971,-0.0809,0.0912,0.0234,-0.0278,-0.1118,0.0764,0.1357,0.0962,0.116,0.2222,-0.0974,-0.0412,0.1753,0.1406,-0.0213,-0.0485,0.055,-0.0168,0.166,-0.2522,0.0545,-0.0637,0.0657,-0.0786,0.1337,-0.1115,-0.0833,0.1007]),
};

// ---------------------------------------------------------------------------
// MJCF Parser (server-side, no DOMParser - use simple XML parsing)
// ---------------------------------------------------------------------------

function _normalize(v) {
  const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return n < 1e-12 ? v : v.map(x => x / n);
}

function _cross(a, b) {
  return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
}

function _dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

function _getRotationQuat(from, to) {
  const u = _normalize(from), v = _normalize(to);
  const d = _dot(u, v);
  if (d > 1 - 1e-6) return [0, 0, 0, 1];
  if (d < 1e-6 - 1) {
    let axis = _cross([1,0,0], u);
    if (_dot(axis, axis) < 1e-6) axis = _cross([0,1,0], u);
    axis = _normalize(axis);
    return [axis[0], axis[1], axis[2], 0];
  }
  const c = _cross(u, v);
  const s = Math.sqrt((1 + d) * 2), inv = 1 / s;
  const q = [c[0]*inv, c[1]*inv, c[2]*inv, 0.5*s];
  const qn = Math.sqrt(q.reduce((s2, x) => s2 + x * x, 0));
  return q.map(x => x / qn);
}

function _quatRotate(q, v) {
  const qv = [q[0],q[1],q[2]], qw = q[3];
  const t = [2*(qv[1]*v[2]-qv[2]*v[1]), 2*(qv[2]*v[0]-qv[0]*v[2]), 2*(qv[0]*v[1]-qv[1]*v[0])];
  return [v[0]+qw*t[0]+qv[1]*t[2]-qv[2]*t[1],
          v[1]+qw*t[1]+qv[2]*t[0]-qv[0]*t[2],
          v[2]+qw*t[2]+qv[0]*t[1]-qv[1]*t[0]];
}

function _mat33ToQuat(cols) {
  const [m00,m10,m20] = cols[0], [m01,m11,m21] = cols[1], [m02,m12,m22] = cols[2];
  const tr = m00 + m11 + m22;
  let x, y, z, w;
  if (tr >= 0) {
    const h = Math.sqrt(tr + 1); w = 0.5*h; const f = 0.5/h;
    x = (m21-m12)*f; y = (m02-m20)*f; z = (m10-m01)*f;
  } else {
    let i = 0;
    if (m11 > m00) i = 1;
    if (m22 > [m00,m11,m22][i]) i = 2;
    if (i === 0) {
      const h = Math.sqrt(m00-m11-m22+1); x = 0.5*h; const f = 0.5/h;
      y = (m01+m10)*f; z = (m20+m02)*f; w = (m21-m12)*f;
    } else if (i === 1) {
      const h = Math.sqrt(m11-m22-m00+1); y = 0.5*h; const f = 0.5/h;
      z = (m12+m21)*f; x = (m01+m10)*f; w = (m02-m20)*f;
    } else {
      const h = Math.sqrt(m22-m00-m11+1); z = 0.5*h; const f = 0.5/h;
      x = (m20+m02)*f; y = (m12+m21)*f; w = (m10-m01)*f;
    }
  }
  const qr = [x,y,z,w], n = Math.sqrt(qr.reduce((s,v)=>s+v*v,0));
  return qr.map(v => v/n);
}

function _computeJointFrame(jointAxes) {
  const axisMap = [0, 1, 2];
  const n = jointAxes.length;
  if (n === 0) return { q: [0,0,0,1], axisMap };
  if (n === 1) return { q: _getRotationQuat([1,0,0], jointAxes[0]), axisMap };

  const Q = _getRotationQuat(jointAxes[0], [1,0,0]);
  const b = _normalize(_quatRotate(Q, jointAxes[1]));

  if (n === 2) {
    if (Math.abs(_dot(b,[0,1,0])) > Math.abs(_dot(b,[0,0,1]))) {
      axisMap[1] = 1;
      const c = _normalize(_cross(jointAxes[0], jointAxes[1]));
      return { q: _mat33ToQuat([_normalize(jointAxes[0]), _normalize(jointAxes[1]), c]), axisMap };
    } else {
      axisMap[1] = 2; axisMap[2] = 1;
      const c = _normalize(_cross(jointAxes[1], jointAxes[0]));
      return { q: _mat33ToQuat([_normalize(jointAxes[0]), c, _normalize(jointAxes[1])]), axisMap };
    }
  }
  if (Math.abs(_dot(b,[0,1,0])) > Math.abs(_dot(b,[0,0,1]))) {
    axisMap[1] = 1; axisMap[2] = 2;
    return { q: _mat33ToQuat([_normalize(jointAxes[0]), _normalize(jointAxes[1]), _normalize(jointAxes[2])]), axisMap };
  } else {
    axisMap[1] = 2; axisMap[2] = 1;
    return { q: _mat33ToQuat([_normalize(jointAxes[0]), _normalize(jointAxes[2]), _normalize(jointAxes[1])]), axisMap };
  }
}

function _parseVec(s, n) {
  if (!s) return null;
  const parts = s.trim().split(/\s+/).map(Number);
  return n ? parts.slice(0, n) : parts;
}

// Minimal XML parser for MJCF (server-side, no DOMParser)
function parseXMLString(text) {
  // Simple recursive descent XML parser
  let pos = 0;
  function skipWhitespace() { while (pos < text.length && /\s/.test(text[pos])) pos++; }
  function parseComment() {
    if (text.substr(pos, 4) === '<!--') {
      pos = text.indexOf('-->', pos) + 3;
      if (pos === 2) pos = text.length;
    }
  }
  function parseProcInstr() {
    if (text.substr(pos, 2) === '<?') {
      pos = text.indexOf('?>', pos) + 2;
      if (pos === 1) pos = text.length;
    }
  }
  function parseNode() {
    skipWhitespace();
    while (text.substr(pos, 4) === '<!--') { parseComment(); skipWhitespace(); }
    while (text.substr(pos, 2) === '<?') { parseProcInstr(); skipWhitespace(); }
    if (text[pos] !== '<' || text[pos+1] === '/') return null;
    pos++; // skip <
    let tagName = '';
    while (pos < text.length && /[a-zA-Z0-9_:-]/.test(text[pos])) tagName += text[pos++];
    const attrs = {};
    while (true) {
      skipWhitespace();
      if (text[pos] === '/' && text[pos+1] === '>') { pos += 2; return { tag: tagName, attrs, children: [] }; }
      if (text[pos] === '>') { pos++; break; }
      let aname = '';
      while (pos < text.length && /[a-zA-Z0-9_:-]/.test(text[pos])) aname += text[pos++];
      skipWhitespace();
      if (text[pos] === '=') {
        pos++;
        skipWhitespace();
        const quote = text[pos]; pos++;
        let val = '';
        while (pos < text.length && text[pos] !== quote) val += text[pos++];
        pos++; // skip closing quote
        attrs[aname] = val;
      }
    }
    const children = [];
    while (true) {
      skipWhitespace();
      while (text.substr(pos, 4) === '<!--') { parseComment(); skipWhitespace(); }
      if (text.substr(pos, 2) === '</') {
        pos = text.indexOf('>', pos) + 1;
        break;
      }
      if (pos >= text.length) break;
      if (text[pos] === '<') {
        const child = parseNode();
        if (child) children.push(child);
      } else {
        // text content - skip
        while (pos < text.length && text[pos] !== '<') pos++;
      }
    }
    return { tag: tagName, attrs, children };
  }
  return parseNode();
}

// Query helpers for the simple XML tree
function xmlQueryOne(node, tagName) {
  if (!node || !node.children) return null;
  for (const c of node.children) {
    if (c.tag === tagName) return c;
  }
  return null;
}

function xmlQueryAll(node, tagName) {
  if (!node || !node.children) return [];
  return node.children.filter(c => c.tag === tagName);
}

function xmlQueryDeep(node, tagName) {
  if (!node || !node.children) return null;
  for (const c of node.children) {
    if (c.tag === tagName) return c;
    const found = xmlQueryDeep(c, tagName);
    if (found) return found;
  }
  return null;
}

function parseMJCF(xmlText, opts = {}) {
  const root = parseXMLString(xmlText);

  let fixedBodies = opts.fixedBodies;
  if (!fixedBodies) {
    fixedBodies = new Set();
    const scan = (el, isRoot) => {
      if (!isRoot) {
        const hinges = xmlQueryAll(el, 'joint').filter(
          j => (j.attrs.type || 'hinge') !== 'free');
        const freeJ = xmlQueryOne(el, 'freejoint');
        if (!hinges.length && !freeJ) fixedBodies.add(el.attrs.name);
      }
      for (const child of xmlQueryAll(el, 'body')) scan(child, false);
    };
    const worldbody = xmlQueryDeep(root, 'worldbody');
    for (const b of xmlQueryAll(worldbody, 'body')) scan(b, true);
  }

  const actuatorMap = {};
  const actuatorOrder = [];
  const actuatorSec = xmlQueryDeep(root, 'actuator');
  if (actuatorSec) {
    for (const mot of xmlQueryAll(actuatorSec, 'motor')) {
      const jname = mot.attrs.joint;
      if (!jname) continue;
      const gear = parseFloat(mot.attrs.gear || '1');
      const frc = _parseVec(mot.attrs.actuatorfrcrange, 2);
      const maxForce = frc ? Math.max(Math.abs(frc[0]), Math.abs(frc[1])) : gear;
      actuatorMap[jname] = { gear, maxForce };
      actuatorOrder.push(jname);
    }
  }

  const bodies = [], joints = [], fixedJoints = [];

  function processBody(el, parentName, parentWorldPos) {
    const name = el.attrs.name;
    const localPos = _parseVec(el.attrs.pos, 3) || [0,0,0];
    const worldPos = localPos.map((v, i) => parentWorldPos[i] + v);

    const geoms = [];
    for (const ge of xmlQueryAll(el, 'geom')) {
      const g = { name: ge.attrs.name || name, type: ge.attrs.type || 'sphere' };
      g.pos = _parseVec(ge.attrs.pos, 3) || [0,0,0];
      if (ge.attrs.size) g.size = _parseVec(ge.attrs.size);
      if (g.type === 'sphere') g.radius = g.size[0];
      else if (g.type === 'capsule') {
        g.radius = g.size[0];
        if (ge.attrs.fromto) g.fromto = _parseVec(ge.attrs.fromto, 6);
      } else if (g.type === 'box') {
        g.halfExtents = g.size.slice(0, 3);
      } else if (g.type === 'cylinder') {
        g.radius = g.size[0];
        if (g.size.length > 1) g.halfHeight = g.size[1];
        if (ge.attrs.fromto) {
          g.fromto = _parseVec(ge.attrs.fromto, 6);
          const ft = g.fromto;
          g.halfHeight = Math.sqrt((ft[3]-ft[0])**2+(ft[4]-ft[1])**2+(ft[5]-ft[2])**2) / 2;
        }
        if (!g.halfHeight) g.halfHeight = 0.1;
      }
      g.density = parseFloat(ge.attrs.density || '1000');
      geoms.push(g);
    }

    const PI = Math.PI;
    function geomMassAndCenter(g) {
      let volume = 0, center = g.pos.slice();
      if (g.type === 'sphere') {
        volume = (4/3) * PI * g.radius ** 3;
      } else if (g.type === 'capsule' && g.fromto) {
        const ft = g.fromto, r = g.radius;
        const halfH = Math.sqrt((ft[3]-ft[0])**2+(ft[4]-ft[1])**2+(ft[5]-ft[2])**2) / 2;
        volume = PI * r*r * (2*halfH) + (4/3) * PI * r**3;
        center = [(ft[0]+ft[3])/2, (ft[1]+ft[4])/2, (ft[2]+ft[5])/2];
      } else if (g.type === 'capsule') {
        const r = g.radius, hh = g.size.length > 1 ? g.size[1] : 0.1;
        volume = PI * r*r * (2*hh) + (4/3) * PI * r**3;
      } else if (g.type === 'box') {
        const he = g.halfExtents || g.size.slice(0,3);
        volume = 8 * he[0] * he[1] * he[2];
      } else if (g.type === 'cylinder') {
        const r = g.radius, hh = g.halfHeight || 0.1;
        volume = PI * r*r * (2*hh);
        if (g.fromto) {
          const ft = g.fromto;
          center = [(ft[0]+ft[3])/2, (ft[1]+ft[4])/2, (ft[2]+ft[5])/2];
        }
      }
      return { mass: volume * g.density, center, volume };
    }

    function capsuleInertia(g) {
      const ft = g.fromto, r = g.radius;
      const dx = ft[3]-ft[0], dy = ft[4]-ft[1], dz = ft[5]-ft[2];
      const L = Math.sqrt(dx*dx+dy*dy+dz*dz);
      const halfH = L / 2;
      const cylVol = PI * r*r * L;
      const cylM = cylVol * g.density;
      const sphVol = (4/3) * PI * r**3;
      const sphM = sphVol * g.density;
      const cylIaxial = cylM * r*r / 2;
      const cylIperp = cylM * (3*r*r + L*L) / 12;
      const sphIaxial = 2 * sphM * r*r / 5;
      const sphIperp = sphIaxial + sphM * (3*r/8 + halfH)**2;
      const Iaxial = cylIaxial + sphIaxial;
      const Iperp = cylIperp + sphIperp;
      if (L < 1e-10) return [Iperp, Iperp, Iperp];
      const ax = [dx/L, dy/L, dz/L];
      return [
        Iperp + (Iaxial - Iperp) * ax[0]*ax[0],
        Iperp + (Iaxial - Iperp) * ax[1]*ax[1],
        Iperp + (Iaxial - Iperp) * ax[2]*ax[2],
      ];
    }

    let totalMass = 0;
    let com = [0, 0, 0];
    const geomData = geoms.map(g => {
      const { mass, center } = geomMassAndCenter(g);
      return { g, mass, center };
    });
    for (const { mass, center } of geomData) {
      com[0] += mass * center[0];
      com[1] += mass * center[1];
      com[2] += mass * center[2];
      totalMass += mass;
    }
    if (totalMass > 0) com = com.map(c => c / totalMass);

    let inertia = [0, 0, 0];
    for (const { g, mass, center } of geomData) {
      let Ii;
      if (g.type === 'sphere') {
        const I = (2/5) * mass * g.radius**2;
        Ii = [I, I, I];
      } else if (g.type === 'capsule' && g.fromto) {
        Ii = capsuleInertia(g);
      } else if (g.type === 'box') {
        const he = g.halfExtents || g.size.slice(0,3);
        const a=2*he[0], b2=2*he[1], c=2*he[2];
        Ii = [mass*(b2*b2+c*c)/12, mass*(a*a+c*c)/12, mass*(a*a+b2*b2)/12];
      } else if (g.type === 'cylinder') {
        const r = g.radius, hh = g.halfHeight || 0.015;
        const Iaxial = mass * r*r / 2;
        const Iperp2 = mass * (3*r*r + (2*hh)**2) / 12;
        if (g.fromto) {
          const ft = g.fromto, L = 2*hh;
          const ddx = ft[3]-ft[0], ddy = ft[4]-ft[1], ddz = ft[5]-ft[2];
          if (L > 1e-10) {
            const ax = [ddx/L, ddy/L, ddz/L];
            Ii = [Iperp2 + (Iaxial-Iperp2)*ax[0]*ax[0],
                  Iperp2 + (Iaxial-Iperp2)*ax[1]*ax[1],
                  Iperp2 + (Iaxial-Iperp2)*ax[2]*ax[2]];
          } else { Ii = [Iperp2, Iperp2, Iaxial]; }
        } else { Ii = [Iperp2, Iperp2, Iaxial]; }
      } else {
        Ii = [0, 0, 0];
      }
      const ddx2 = center[0]-com[0], ddy2 = center[1]-com[1], ddz2 = center[2]-com[2];
      inertia[0] += Ii[0] + mass*(ddy2*ddy2+ddz2*ddz2);
      inertia[1] += Ii[1] + mass*(ddx2*ddx2+ddz2*ddz2);
      inertia[2] += Ii[2] + mass*(ddx2*ddx2+ddy2*ddy2);
    }

    bodies.push({ name, parent: parentName, pos: worldPos, localPos, geoms,
                   mass: totalMass, inertia, com });

    const jointEls = xmlQueryAll(el, 'joint').filter(
      j => (j.attrs.type || 'hinge') !== 'free' && j.tag !== 'freejoint');

    if (fixedBodies.has(name)) {
      fixedJoints.push({ name: name+'_fixed', parent_body: parentName, child_body: name, localPos0: localPos });
    } else if (jointEls.length && parentName !== null) {
      const axesData = [], jointAxes = [];
      for (const je of jointEls) {
        const axis = _parseVec(je.attrs.axis || '1 0 0', 3);
        jointAxes.push(axis);
        const rng = _parseVec(je.attrs.range || '-3.14159 3.14159', 2);
        const jname = je.attrs.name;
        const act = actuatorMap[jname] || { gear: 100, maxForce: 100 };
        axesData.push({
          name: jname, mjcf_axis: axis,
          stiffness: parseFloat(je.attrs.stiffness || '0'),
          damping: parseFloat(je.attrs.damping || '0'),
          maxForce: act.maxForce,
          range: rng,
          armature: parseFloat(je.attrs.armature || '0'),
        });
      }

      const { q, axisMap } = _computeJointFrame(jointAxes);
      const localRot = [q[3], q[0], q[1], q[2]]; // wxyz

      joints.push({
        name: jointEls.length > 1 ? jointEls[0].attrs.name.replace(/_[^_]+$/, '') : jointEls[0].attrs.name,
        parent_body: parentName, child_body: name,
        axes: axesData,
        axisMap: axisMap.slice(0, jointEls.length),
        localPos0: localPos,
        localRot,
        jointType: jointEls.length > 1 ? 'spherical' : 'revolute',
      });
    }

    for (const child of xmlQueryAll(el, 'body'))
      processBody(child, name, worldPos);
  }

  const worldbody = xmlQueryDeep(root, 'worldbody');
  const pelvis = xmlQueryAll(worldbody, 'body')[0];
  const pelvisPos = _parseVec(pelvis.attrs.pos, 3) || [0,0,0];
  processBody(pelvis, null, pelvisPos);

  const dofInfo = [];
  for (const actName of actuatorOrder) {
    for (const jdata of joints) {
      for (let ai = 0; ai < jdata.axes.length; ai++) {
        if (jdata.axes[ai].name === actName) {
          dofInfo.push({
            joint_name: jdata.name, axis_name: actName,
            physx_axis: jdata.axisMap[ai], child_body: jdata.child_body,
          });
          break;
        }
      }
    }
  }

  const bodyNames = bodies.map(b => b.name);
  const fk_parent_indices = bodies.map(b => b.parent === null ? -1 : bodyNames.indexOf(b.parent));
  const fk_local_translations = bodies.map(b => b.localPos);
  const fk_local_rotations = bodies.map(() => [0, 0, 0, 1]);

  const kinematicJoints = [];
  let dofIdx = 0;
  for (let bi = 1; bi < bodies.length; bi++) {
    const bname = bodies[bi].name;
    const jdata = joints.find(j => j.child_body === bname);
    const fjdata = fixedJoints.find(j => j.child_body === bname);
    if (jdata) {
      const nDofs = jdata.axes.length;
      kinematicJoints.push({
        name: jdata.name, child_body: bname,
        type: nDofs > 1 ? 'SPHERICAL' : 'HINGE',
        dof_idx: dofIdx, dof_dim: nDofs,
        axis: nDofs === 1 ? jdata.axes[0].mjcf_axis : undefined,
      });
      dofIdx += nDofs;
    } else if (fjdata) {
      kinematicJoints.push({ name: fjdata.name, child_body: bname, type: 'FIXED', dof_idx: dofIdx, dof_dim: 0 });
    } else {
      kinematicJoints.push({ name: bname, child_body: bname, type: 'FIXED', dof_idx: dofIdx, dof_dim: 0 });
    }
  }

  return {
    bodies, joints, fixedJoints, actuatorOrder, dofInfo,
    fk_parent_indices, fk_local_translations, fk_local_rotations,
    kinematicJoints,
    act_dim: actuatorOrder.length,
  };
}

// ---------------------------------------------------------------------------
// ONNX metadata extraction
// ---------------------------------------------------------------------------
function extractOnnxMetadata(buffer) {
  const bytes = new Uint8Array(buffer);
  const sentinel = 'mimickit_config';
  const sentinelBytes = new TextEncoder().encode(sentinel);

  for (let i = 0; i < bytes.length - sentinelBytes.length - 10; i++) {
    let match = true;
    for (let j = 0; j < sentinelBytes.length; j++) {
      if (bytes[i + j] !== sentinelBytes[j]) { match = false; break; }
    }
    if (!match) continue;

    const searchStart = i + sentinelBytes.length;
    for (let k = searchStart; k < Math.min(searchStart + 20, bytes.length); k++) {
      if (bytes[k] === 0x12) {
        let len = 0, shift = 0, pos = k + 1;
        while (pos < bytes.length) {
          const b = bytes[pos++];
          len |= (b & 0x7f) << shift;
          shift += 7;
          if ((b & 0x80) === 0) break;
        }
        if (len > 0 && pos + len <= bytes.length) {
          const jsonStr = new TextDecoder().decode(bytes.slice(pos, pos + len));
          try {
            return JSON.parse(jsonStr);
          } catch(e) {
            console.log('Found sentinel but JSON parse failed:', e.message);
          }
        }
        break;
      }
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Quaternion math (same as client, for observation building)
// ---------------------------------------------------------------------------
function quatRotateVec(q, v) {
  const [qx, qy, qz, qw] = q;
  const [vx, vy, vz] = v;
  const tx = 2 * (qy*vz - qz*vy);
  const ty = 2 * (qz*vx - qx*vz);
  const tz = 2 * (qx*vy - qy*vx);
  return [
    vx + qw*tx + (qy*tz - qz*ty),
    vy + qw*ty + (qz*tx - qx*tz),
    vz + qw*tz + (qx*ty - qy*tx)
  ];
}

function quatMul(a, b) {
  const [ax,ay,az,aw] = a, [bx,by,bz,bw] = b;
  return [
    aw*bx + ax*bw + ay*bz - az*by,
    aw*by - ax*bz + ay*bw + az*bx,
    aw*bz + ax*by - ay*bx + az*bw,
    aw*bw - ax*bx - ay*by - az*bz
  ];
}

function quatNorm(q) {
  const [x,y,z,w] = q;
  const len = Math.sqrt(x*x + y*y + z*z + w*w) || 1;
  return [x/len, y/len, z/len, w/len];
}

function axisAngleToQuat(axis, angle) {
  const ha = angle * 0.5;
  const s = Math.sin(ha), c = Math.cos(ha);
  const len = Math.sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]) || 1;
  return quatNorm([axis[0]/len*s, axis[1]/len*s, axis[2]/len*s, c]);
}

function expMapToQuat(ex, ey, ez) {
  const angle = Math.sqrt(ex*ex + ey*ey + ez*ez);
  if (angle < 1e-5) return [0, 0, 0, 1];
  return axisAngleToQuat([ex/angle, ey/angle, ez/angle], angle);
}

function calcHeading(q) {
  const rotDir = quatRotateVec(q, [1, 0, 0]);
  return Math.atan2(rotDir[1], rotDir[0]);
}

function calcHeadingQuatInv(q) {
  const heading = calcHeading(q);
  return axisAngleToQuat([0, 0, 1], -heading);
}

function quatToTanNorm(q) {
  const tan = quatRotateVec(q, [1, 0, 0]);
  const norm = quatRotateVec(q, [0, 0, 1]);
  return [tan[0], tan[1], tan[2], norm[0], norm[1], norm[2]];
}

function quatFromTwoVec(from, to) {
  const [fx,fy,fz] = from, [tx,ty,tz] = to;
  const dot = fx*tx + fy*ty + fz*tz;
  if (dot > 0.999999) return [0,0,0,1];
  if (dot < -0.999999) {
    let ax = [0, -fz, fy];
    if (Math.abs(fx) > 0.9) ax = [fz, 0, -fx];
    const len = Math.sqrt(ax[0]*ax[0]+ax[1]*ax[1]+ax[2]*ax[2]);
    return [ax[0]/len, ax[1]/len, ax[2]/len, 0];
  }
  const cx = fy*tz - fz*ty, cy = fz*tx - fx*tz, cz = fx*ty - fy*tx;
  const w = 1 + dot;
  const len = Math.sqrt(cx*cx + cy*cy + cz*cz + w*w);
  return [cx/len, cy/len, cz/len, w/len];
}

function gaussianRandom() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// ---------------------------------------------------------------------------
// Per-player humanoid state
// ---------------------------------------------------------------------------
class PlayerHumanoid {
  constructor(id, px, physics, pxScene, material, humanoidData, spawnIndex) {
    this.id = id;
    this.name = 'Player ' + (spawnIndex + 1);
    this.color = PLAYER_COLORS[spawnIndex % PLAYER_COLORS.length];

    // Input state from client
    this.moveDir = [1, 0];
    this.faceDir = [1, 0];
    this.speed = 1.5;
    this.actionSlash = false;
    this.actionKick = false;
    this.actionBlock = false;
    this.actionJump = false;

    this.resetGraceFrames = RESET_GRACE;
    this.physStepCount = 0;
    this.currentAction = null;

    // Build articulation
    this.links = [];
    this.origDrives = [];
    this._buildArticulation(px, physics, pxScene, material, humanoidData, spawnIndex);
  }

  _buildArticulation(px, physics, pxScene, material, humanoidData, spawnIndex) {
    const E_TWIST = px.PxArticulationAxisEnum.eTWIST;
    const E_SWING1 = px.PxArticulationAxisEnum.eSWING1;
    const E_SWING2 = px.PxArticulationAxisEnum.eSWING2;
    const E_LIMITED = px.PxArticulationMotionEnum.eLIMITED;
    const E_LOCKED = px.PxArticulationMotionEnum.eLOCKED;
    const E_SPHERICAL = px.PxArticulationJointTypeEnum.eSPHERICAL;
    const E_REVOLUTE = px.PxArticulationJointTypeEnum.eREVOLUTE;
    const E_FIX = px.PxArticulationJointTypeEnum.eFIX;
    const E_FORCE = px.PxArticulationDriveTypeEnum.eFORCE;
    const SHAPE_FLAGS_VAL = px.PxShapeFlagEnum.eSCENE_QUERY_SHAPE | px.PxShapeFlagEnum.eSIMULATION_SHAPE;

    this.axisEnums = [E_TWIST, E_SWING1, E_SWING2];
    this.E_FORCE = E_FORCE;

    const shapeFlags = new px.PxShapeFlags(SHAPE_FLAGS_VAL);
    const rbext = px.PxRigidBodyExt.prototype;

    const art = physics.createArticulationReducedCoordinate();
    art.setSolverIterationCounts(4, 0);
    if (typeof art.setSleepThreshold === 'function') art.setSleepThreshold(5e-5);
    if (typeof art.setStabilizationThreshold === 'function') art.setStabilizationThreshold(1e-5);
    art.setArticulationFlag(px.PxArticulationFlagEnum.eDISABLE_SELF_COLLISION, false);

    this.articulation = art;

    // Spawn offset: spread players in a line
    // Random spawn within a radius
    const spawnRadius = 5;
    const angle = Math.random() * Math.PI * 2;
    const dist = Math.sqrt(Math.random()) * spawnRadius;
    const spawnX = Math.cos(angle) * dist;
    const spawnY = Math.sin(angle) * dist;

    const bodyLinkMap = {};
    for (const body of humanoidData.bodies) {
      const wp = body.pos;
      const zOff = humanoidData.tpose_pelvis_z || humanoidData.pelvis_z;
      const pose = new px.PxTransform(
        new px.PxVec3(wp[0] + spawnX, wp[1] + spawnY, wp[2] + zOff),
        new px.PxQuat(0, 0, 0, 1)
      );

      const parentLink = body.parent ? bodyLinkMap[body.parent] : null;
      const link = art.createLink(parentLink, pose);

      for (const geom of body.geoms) {
        let shape = null;
        if (geom.type === 'sphere') {
          const g = new px.PxSphereGeometry(geom.radius);
          shape = physics.createShape(g, material, true, shapeFlags);
          const lp = geom.pos;
          shape.setLocalPose(new px.PxTransform(
            new px.PxVec3(lp[0], lp[1], lp[2]),
            new px.PxQuat(0, 0, 0, 1)
          ));
        } else if (geom.type === 'capsule' && geom.fromto) {
          const ft = geom.fromto;
          const p0 = [ft[0], ft[1], ft[2]];
          const p1 = [ft[3], ft[4], ft[5]];
          const dx = p1[0]-p0[0], dy = p1[1]-p0[1], dz = p1[2]-p0[2];
          const len = Math.sqrt(dx*dx+dy*dy+dz*dz);
          const halfH = Math.max(len / 2, 0.001);
          const g = new px.PxCapsuleGeometry(geom.radius, halfH);
          shape = physics.createShape(g, material, true, shapeFlags);
          const mid = [(p0[0]+p1[0])/2, (p0[1]+p1[1])/2, (p0[2]+p1[2])/2];
          const dir = len > 0.001 ? [dx/len, dy/len, dz/len] : [1,0,0];
          const q = quatFromTwoVec([1,0,0], dir);
          shape.setLocalPose(new px.PxTransform(
            new px.PxVec3(mid[0], mid[1], mid[2]),
            new px.PxQuat(q[0], q[1], q[2], q[3])
          ));
        } else if (geom.type === 'box') {
          const he = geom.halfExtents;
          const g = new px.PxBoxGeometry(he[0], he[1], he[2]);
          shape = physics.createShape(g, material, true, shapeFlags);
          const lp = geom.pos;
          shape.setLocalPose(new px.PxTransform(
            new px.PxVec3(lp[0], lp[1], lp[2]),
            new px.PxQuat(0, 0, 0, 1)
          ));
        } else if (geom.type === 'cylinder') {
          // Approximate cylinder with capsule (simpler for server)
          const r = geom.radius;
          const hh = geom.halfHeight || 0.015;
          const g = new px.PxCapsuleGeometry(r, hh);
          shape = physics.createShape(g, material, true, shapeFlags);
          if (geom.fromto) {
            const ft = geom.fromto;
            const p0 = [ft[0], ft[1], ft[2]], p1 = [ft[3], ft[4], ft[5]];
            const mid = [(p0[0]+p1[0])/2, (p0[1]+p1[1])/2, (p0[2]+p1[2])/2];
            const dx = p1[0]-p0[0], dy = p1[1]-p0[1], dz = p1[2]-p0[2];
            const len = Math.sqrt(dx*dx+dy*dy+dz*dz);
            const dir = len > 0.001 ? [dx/len, dy/len, dz/len] : [0,0,1];
            const q = quatFromTwoVec([1,0,0], dir);
            shape.setLocalPose(new px.PxTransform(
              new px.PxVec3(mid[0], mid[1], mid[2]),
              new px.PxQuat(q[0], q[1], q[2], q[3])
            ));
          } else {
            const lp = geom.pos || [0,0,0];
            const q = quatFromTwoVec([1,0,0], [0,0,1]);
            shape.setLocalPose(new px.PxTransform(
              new px.PxVec3(lp[0], lp[1], lp[2]),
              new px.PxQuat(q[0], q[1], q[2], q[3])
            ));
          }
        }
        if (shape) {
          shape.setSimulationFilterData(new px.PxFilterData(2, 3, 0, 0));
          link.attachShape(shape);
        }
      }

      if (body.mass && body.inertia && body.com) {
        link.setMass(body.mass);
        link.setMassSpaceInertiaTensor(new px.PxVec3(body.inertia[0], body.inertia[1], body.inertia[2]));
        link.setCMassLocalPose(new px.PxTransform(
          new px.PxVec3(body.com[0], body.com[1], body.com[2]),
          new px.PxQuat(0, 0, 0, 1)
        ));
      } else if (body.mass) {
        rbext.setMassAndUpdateInertia(link, body.mass);
      } else {
        rbext.updateMassAndInertia(link, 1000);
      }

      link.setAngularDamping(0.01);
      link.setLinearDamping(0.0);
      link.setMaxDepenetrationVelocity(10.0);
      link.setMaxLinearVelocity(1000.0);
      link.setMaxAngularVelocity(1000.0);
      if (typeof link.setSleepThreshold === 'function') link.setSleepThreshold(5e-5);
      if (typeof link.setStabilizationThreshold === 'function') link.setStabilizationThreshold(1e-5);
      if (typeof link.setCfmScale === 'function') link.setCfmScale(0.025);
      try { if (typeof link.setRigidBodyFlag === 'function') link.setRigidBodyFlag(px.PxRigidBodyFlagEnum.eENABLE_GYROSCOPIC_FORCES, true); } catch(e) {}

      bodyLinkMap[body.name] = link;
      this.links.push({ name: body.name, link });
    }

    // Configure joints
    const driveScale = 1.0;
    for (const jdata of humanoidData.joints) {
      const childLink = bodyLinkMap[jdata.child_body];
      const joint = childLink.getInboundJoint();

      if (jdata.jointType === 'spherical') {
        joint.setJointType(E_SPHERICAL);
      } else {
        joint.setJointType(E_REVOLUTE);
      }

      joint.setFrictionCoefficient(0);
      if (typeof joint.setMaxJointVelocity === 'function') joint.setMaxJointVelocity(1000000);

      const lp = jdata.localPos0;
      const lr = jdata.localRot;

      joint.setParentPose(new px.PxTransform(
        new px.PxVec3(lp[0], lp[1], lp[2]),
        new px.PxQuat(lr[1], lr[2], lr[3], lr[0])
      ));
      joint.setChildPose(new px.PxTransform(
        new px.PxVec3(0, 0, 0),
        new px.PxQuat(lr[1], lr[2], lr[3], lr[0])
      ));

      if (jdata.jointType === 'spherical') {
        for (let i = 0; i < jdata.axes.length; i++) {
          const physAxis = jdata.axisMap[i];
          const axE = this.axisEnums[physAxis];
          joint.setMotion(axE, E_LIMITED);
          const ax = jdata.axes[i];
          joint.setLimitParams(axE, new px.PxArticulationLimit(ax.range[0], ax.range[1]));
          joint.setDriveParams(axE, new px.PxArticulationDrive(
            ax.stiffness * driveScale, ax.damping * driveScale, ax.maxForce, E_FORCE
          ));
          if (ax.armature !== undefined) joint.setArmature(axE, ax.armature);
          this.origDrives.push({joint, axisEnum: axE, stiffness: ax.stiffness * driveScale, damping: ax.damping * driveScale, maxForce: ax.maxForce});
        }
        for (let i = jdata.axes.length; i < 3; i++) {
          joint.setMotion(this.axisEnums[i], E_LOCKED);
        }
      } else {
        joint.setMotion(E_TWIST, E_LIMITED);
        const ax = jdata.axes[0];
        joint.setLimitParams(E_TWIST, new px.PxArticulationLimit(ax.range[0], ax.range[1]));
        joint.setDriveParams(E_TWIST, new px.PxArticulationDrive(
          ax.stiffness * driveScale, ax.damping * driveScale, ax.maxForce, E_FORCE
        ));
        if (ax.armature !== undefined) joint.setArmature(E_TWIST, ax.armature);
        this.origDrives.push({joint, axisEnum: E_TWIST, stiffness: ax.stiffness * driveScale, damping: ax.damping * driveScale, maxForce: ax.maxForce});
        joint.setMotion(E_SWING1, E_LOCKED);
        joint.setMotion(E_SWING2, E_LOCKED);
      }
    }

    // Fixed joints
    for (const fj of humanoidData.fixedJoints) {
      const childLink = bodyLinkMap[fj.child_body];
      const joint = childLink.getInboundJoint();
      joint.setJointType(E_FIX);
      const lp = fj.localPos0;
      joint.setParentPose(new px.PxTransform(
        new px.PxVec3(lp[0], lp[1], lp[2]),
        new px.PxQuat(0, 0, 0, 1)
      ));
      joint.setChildPose(new px.PxTransform(
        new px.PxVec3(0, 0, 0),
        new px.PxQuat(0, 0, 0, 1)
      ));
    }

    pxScene.addArticulation(art);
    this._applyInitPose(px, humanoidData, spawnX, spawnY);
  }

  _applyInitPose(px, humanoidData, spawnX, spawnY) {
    if (!humanoidData.init_root_pos || !humanoidData.init_dof_pos) return;

    const rp = humanoidData.init_root_pos;
    const rq = humanoidData.init_root_rot_quat;
    const initDof = humanoidData.init_dof_pos;

    this.articulation.setRootGlobalPose(new px.PxTransform(
      new px.PxVec3(rp[0] + spawnX, rp[1] + spawnY, rp[2]),
      new px.PxQuat(rq[0], rq[1], rq[2], rq[3])), true);

    const bodyLinkMap = {};
    for (const l of this.links) bodyLinkMap[l.name] = l.link;

    for (let i = 0; i < humanoidData.dofInfo.length; i++) {
      const dof = humanoidData.dofInfo[i];
      const cl = bodyLinkMap[dof.child_body];
      if (!cl) continue;
      try {
        cl.getInboundJoint().setJointPosition(this.axisEnums[dof.physx_axis], initDof[i]);
        cl.getInboundJoint().setJointVelocity(this.axisEnums[dof.physx_axis], 0);
      } catch(e) {}
    }

    const initFlags = new px.PxArticulationCacheFlags(px.PxArticulationCacheFlagEnum.eALL);
    const initCache = this.articulation.createCache();
    this.articulation.copyInternalStateToCache(initCache, initFlags);
    this.articulation.applyCache(initCache, initFlags, true);

    // Set drive targets to init pose
    for (let i = 0; i < humanoidData.dofInfo.length; i++) {
      const dof = humanoidData.dofInfo[i];
      const childLink = bodyLinkMap[dof.child_body];
      if (!childLink) continue;
      try {
        childLink.getInboundJoint().setDriveTarget(this.axisEnums[dof.physx_axis], initDof[i]);
      } catch(e) {}
    }
  }

  buildObservation(humanoidData) {
    const obs = new Float32Array(humanoidData.obs_dim);
    const bodyLinkMap = {};
    for (const l of this.links) bodyLinkMap[l.name] = l.link;

    const rootPose = this.links[0].link.getGlobalPose();
    const rpx = rootPose.get_p();
    const rqx = rootPose.get_q();

    const rootPos = [rpx.get_x(), rpx.get_y(), rpx.get_z()];
    const rootRot = [rqx.get_x(), rqx.get_y(), rqx.get_z(), rqx.get_w()];

    const rv = this.links[0].link.getLinearVelocity();
    const rootVel = [rv.get_x(), rv.get_y(), rv.get_z()];
    const raw = this.links[0].link.getAngularVelocity();
    const rootAngVel = [raw.get_x(), raw.get_y(), raw.get_z()];

    const headingInv = calcHeadingQuatInv(rootRot);

    let idx = 0;
    obs[idx++] = rootPos[2];

    const localRootRot = quatMul(headingInv, rootRot);
    const rootTanNorm = quatToTanNorm(localRootRot);
    for (let i = 0; i < 6; i++) obs[idx++] = rootTanNorm[i];

    const localVel = quatRotateVec(headingInv, rootVel);
    obs[idx++] = localVel[0]; obs[idx++] = localVel[1]; obs[idx++] = localVel[2];

    const localAngVel = quatRotateVec(headingInv, rootAngVel);
    obs[idx++] = localAngVel[0]; obs[idx++] = localAngVel[1]; obs[idx++] = localAngVel[2];

    const dofPositions = new Float32Array(31);
    for (let i = 0; i < humanoidData.dofInfo.length; i++) {
      const dof = humanoidData.dofInfo[i];
      const childLink = bodyLinkMap[dof.child_body];
      if (childLink) {
        try {
          dofPositions[i] = childLink.getInboundJoint().getJointPosition(this.axisEnums[dof.physx_axis]);
        } catch(e) { dofPositions[i] = 0; }
      }
    }

    for (const kj of humanoidData.kinematicJoints) {
      let quat;
      if (kj.type === 'SPHERICAL') {
        quat = expMapToQuat(dofPositions[kj.dof_idx], dofPositions[kj.dof_idx + 1], dofPositions[kj.dof_idx + 2]);
      } else if (kj.type === 'HINGE') {
        quat = axisAngleToQuat(kj.axis, dofPositions[kj.dof_idx]);
      } else {
        quat = [0, 0, 0, 1];
      }
      const tn = quatToTanNorm(quat);
      for (let k = 0; k < 6; k++) obs[idx++] = tn[k];
    }

    for (const dof of humanoidData.dofInfo) {
      try {
        const childLink = bodyLinkMap[dof.child_body];
        if (childLink) {
          obs[idx++] = childLink.getInboundJoint().getJointVelocity(this.axisEnums[dof.physx_axis]);
        } else { obs[idx++] = 0; }
      } catch(e) { obs[idx++] = 0; }
    }

    const keyIds = humanoidData.key_body_ids || [2, 5, 10, 13, 16, 6];
    for (const bid of keyIds) {
      const linkPose = this.links[bid].link.getGlobalPose();
      const lp = linkPose.get_p();
      const bp = [lp.get_x(), lp.get_y(), lp.get_z()];
      const rel = [bp[0] - rootPos[0], bp[1] - rootPos[1], bp[2] - rootPos[2]];
      const localRel = quatRotateVec(headingInv, rel);
      obs[idx++] = localRel[0]; obs[idx++] = localRel[1]; obs[idx++] = localRel[2];
    }

    return obs;
  }

  buildTaskObs() {
    const taskObs = new Float32Array(5);
    const rootPose = this.links[0].link.getGlobalPose();
    const rqx = rootPose.get_q();
    const rootRot = [rqx.get_x(), rqx.get_y(), rqx.get_z(), rqx.get_w()];
    const headingInv = calcHeadingQuatInv(rootRot);

    const localTarDir = quatRotateVec(headingInv, [this.moveDir[0], this.moveDir[1], 0]);
    const localFaceDir = quatRotateVec(headingInv, [-this.faceDir[0], -this.faceDir[1], 0]);

    taskObs[0] = localTarDir[0]; taskObs[1] = localTarDir[1];
    taskObs[2] = this.speed;
    taskObs[3] = localFaceDir[0]; taskObs[4] = localFaceDir[1];
    return taskObs;
  }

  applyActions(action, humanoidData) {
    const bodyLinkMap = {};
    for (const l of this.links) bodyLinkMap[l.name] = l.link;
    const aLow = humanoidData.action_low;
    const aHigh = humanoidData.action_high;

    for (let i = 0; i < humanoidData.dofInfo.length; i++) {
      const dof = humanoidData.dofInfo[i];
      const childLink = bodyLinkMap[dof.child_body];
      if (!childLink) continue;
      try {
        let a = action[i];
        if (aLow && aHigh) a = Math.max(aLow[i], Math.min(aHigh[i], a));
        childLink.getInboundJoint().setDriveTarget(this.axisEnums[dof.physx_axis], a);
      } catch(e) {}
    }
  }

  getLinkPoses() {
    const poses = new Float32Array(this.links.length * 7);
    for (let i = 0; i < this.links.length; i++) {
      const pose = this.links[i].link.getGlobalPose();
      const p = pose.get_p();
      const q = pose.get_q();
      const off = i * 7;
      poses[off]   = p.get_x();
      poses[off+1] = p.get_y();
      poses[off+2] = p.get_z();
      poses[off+3] = q.get_x();
      poses[off+4] = q.get_y();
      poses[off+5] = q.get_z();
      poses[off+6] = q.get_w();
    }
    return poses;
  }

  needsReset() {
    if (this.resetGraceFrames > 0) { this.resetGraceFrames--; return false; }
    try {
      const rz = this.links[0].link.getGlobalPose().get_p().get_z();
      if (isNaN(rz) || Math.abs(rz) > 50) return true;
      // Respawn if fallen (pelvis Z below 0.4)
      if (rz < 0.4) return true;
    } catch(e) { return true; }
    return false;
  }

  destroy(pxScene) {
    try { pxScene.removeArticulation(this.articulation); } catch(e) {}
  }
}

// ---------------------------------------------------------------------------
// ONNX inference helper (sequential per player, batching would require
// dynamic batch support which many exported models don't have)
// ---------------------------------------------------------------------------
function runInferenceForPlayer(sessionHLC, sessionLLC, obs, taskObs, humanoidData, player) {
  const obsDim = humanoidData.obs_dim;
  const latentDim = humanoidData.latent_dim || 64;

  // Check for action latent override
  let actionLatent = null;
  if (player.actionSlash) actionLatent = ACTION_LATENTS.slash;
  else if (player.actionKick) actionLatent = ACTION_LATENTS.kick;
  else if (player.actionBlock) actionLatent = ACTION_LATENTS.block;
  else if (player.actionJump) actionLatent = ACTION_LATENTS.jump;

  if (actionLatent) {
    const llcResult = runSession(sessionLLC, {
      obs: { data: obs, dims: [1, obsDim] },
      latent: { data: actionLatent, dims: [1, latentDim] },
    });
    return llcResult.action;
  } else {
    if (!sessionHLC) return null;
    const hlcResult = runSession(sessionHLC, {
      obs: { data: obs, dims: [1, obsDim] },
      task_obs: { data: taskObs, dims: [1, 5] },
    });
    const z = hlcResult.z;
    const llcResult = runSession(sessionLLC, {
      obs: { data: obs, dims: [1, obsDim] },
      latent: { data: z, dims: [1, latentDim] },
    });
    return llcResult.action;
  }
}

// ---------------------------------------------------------------------------
// PartyKit Server
// ---------------------------------------------------------------------------

/** @implements {Server} */
class PartyServer {
  /** @param {Room} room */
  constructor(room) {
    this.room = room;
    /** @type {Record<string, PlayerHumanoid>} */
    this.players = {};
    this.globalPlayerCount = 0;

    // PhysX
    this.px = null;
    this.pxScene = null;
    this.physics = null;
    this.material = null;
    this.groundActor = null;

    // ONNX sessions (from ort-server.js)
    this.sessionLLC = null;
    this.sessionHLC = null;

    // Humanoid template data
    this.humanoidData = null;

    this.tickCounter = 0;
    this.lastPhysTime = Date.now();
    this.simAccum = 0;

    this.interval = null;
  }

  async onStart() {
    console.log('Server starting...');

    // Initialize PhysX
    this.px = await PhysXInit({
      instantiateWasm: (imports, callback) => {
        const instance = new WebAssembly.Instance(physxWasm, imports);
        callback(instance);
        return instance.exports;
      }
    });

    const px = this.px;
    const version = px.PHYSICS_VERSION;
    const allocator = new px.PxDefaultAllocator();
    const errorCb = new px.PxDefaultErrorCallback();
    const foundation = px.CreateFoundation(version, allocator, errorCb);
    const tolerances = new px.PxTolerancesScale();
    this.physics = px.CreatePhysics(version, foundation, tolerances);

    const sceneDesc = new px.PxSceneDesc(tolerances);
    sceneDesc.set_gravity(new px.PxVec3(0, 0, -9.81));
    sceneDesc.set_cpuDispatcher(px.DefaultCpuDispatcherCreate(0));
    sceneDesc.set_filterShader(px.DefaultFilterShader());
    if (px.PxSolverTypeEnum && px.PxSolverTypeEnum.ePGS !== undefined) {
      sceneDesc.set_solverType(px.PxSolverTypeEnum.ePGS);
    }
    this.pxScene = this.physics.createScene(sceneDesc);

    this.material = this.physics.createMaterial(1.0, 1.0, 0.0);

    // Ground plane
    const SHAPE_FLAGS_VAL = px.PxShapeFlagEnum.eSCENE_QUERY_SHAPE | px.PxShapeFlagEnum.eSIMULATION_SHAPE;
    const groundShapeFlags = new px.PxShapeFlags(SHAPE_FLAGS_VAL);
    const groundGeom = new px.PxBoxGeometry(10, 10, 0.5);
    const groundShape = this.physics.createShape(groundGeom, this.material, true, groundShapeFlags);
    groundShape.setSimulationFilterData(new px.PxFilterData(1, 7, 0, 0));
    this.groundActor = this.physics.createRigidStatic(new px.PxTransform(
      new px.PxVec3(0, 0, -0.5), new px.PxQuat(0, 0, 0, 1)
    ));
    this.groundActor.attachShape(groundShape);
    this.pxScene.addActor(this.groundActor);

    console.log('PhysX initialized');

    // Load ONNX models via our custom ORT wrapper (bypasses ort npm)
    // Fetch ONNX model files from our own static file serving
    const pkHost = this.room.env?.PARTYKIT_HOST || 'swordbrawl.zalo.partykit.dev';
    const pkProto = pkHost.startsWith('localhost') || pkHost.startsWith('127.') ? 'http' : 'https';
    const baseUrl = `${pkProto}://${pkHost}`;

    try {
      await initOrt();
      console.log('ORT WASM module ready');
    } catch(e) {
      console.error('ORT init failed:', e);
    }
    console.log('Fetching ONNX models from:', baseUrl);

    let llcBuf;
    try {
      const llcResp = await fetch(`${baseUrl}/llc_sword_shield.onnx`);
      llcBuf = await llcResp.arrayBuffer();
      console.log('LLC fetched:', llcBuf.byteLength, 'bytes');
    } catch(e) {
      console.error('LLC fetch failed:', e);
    }

    // Extract metadata from LLC binary (before creating session)
    const meta = extractOnnxMetadata(llcBuf);
    if (meta) {
      this.humanoidData = {};
      const fields = ['obs_dim','act_dim','latent_dim','obs_mean','obs_std','a_mean','a_std',
                      'init_dof_pos','init_root_pos','init_root_rot_quat','action_low','action_high',
                      'key_body_ids','global_obs','pelvis_z','tpose_pelvis_z'];
      for (const f of fields) {
        if (meta[f] !== undefined) this.humanoidData[f] = meta[f];
      }
      if (meta.mjcf_xml) {
        const mjcfData = parseMJCF(meta.mjcf_xml);
        Object.assign(this.humanoidData, mjcfData);
        console.log('Parsed MJCF:', this.humanoidData.bodies.length, 'bodies');
      }
      if (this.humanoidData.kinematicJoints) {
        for (const kj of this.humanoidData.kinematicJoints) {
          if (kj.type !== 'SPHERICAL') continue;
          for (let d = 0; d < 3; d++)
            this.humanoidData.dofInfo[kj.dof_idx + d].physx_axis = d;
        }
        for (const jdata of this.humanoidData.joints) {
          if (jdata.jointType === 'spherical' && jdata.axisMap)
            jdata.axisMap = [0, 1, 2];
        }
      }
    }

    try {
      this.sessionLLC = await createSession(new Uint8Array(llcBuf));
      console.log('LLC session created');
    } catch(e) {
      console.error('LLC session creation failed:', e);
    }

    try {
      const hlcResp = await fetch(`${baseUrl}/hlc_steering_v2.onnx`);
      const hlcBuf = await hlcResp.arrayBuffer();
      console.log('HLC fetched:', hlcBuf.byteLength, 'bytes');
      this.sessionHLC = await createSession(new Uint8Array(hlcBuf));
      console.log('HLC session created');
    } catch(e) {
      console.error('HLC session creation failed:', e);
    }

    // Start tick loop (all inference is synchronous WASM calls)
    this.lastPhysTime = Date.now();
    this.interval = setInterval(() => this._tick(), 1000 / TICK_HZ);

    console.log('Server ready');
  }

  _tick() {
    if (!this.pxScene || !this.humanoidData) return;

    const playerIds = Object.keys(this.players);
    if (playerIds.length === 0) return;

    const now = Date.now();
    const realDT = Math.min((now - this.lastPhysTime) / 1000, 0.1);
    this.lastPhysTime = now;
    this.simAccum += realDT;

    let stepsThisFrame = 0;
    const maxSteps = 8;

    while (this.simAccum >= DT && stepsThisFrame < maxSteps) {
      this.simAccum -= DT;
      stepsThisFrame++;

      // Run policy for each player at control frequency
      for (const pid of playerIds) {
        const player = this.players[pid];
        if (!player) continue;
        player.physStepCount++;

        if (player.physStepCount >= NUM_SUBSTEPS) {
          player.physStepCount = 0;

          if (this.sessionLLC) {
            try {
              const obs = player.buildObservation(this.humanoidData);
              const taskObs = player.buildTaskObs();
              const action = runInferenceForPlayer(
                this.sessionHLC, this.sessionLLC, obs, taskObs, this.humanoidData, player);
              if (action) {
                player.currentAction = action;
                player.applyActions(action, this.humanoidData);
              }
            } catch(e) {
              if (!this._infErrLogged) { this._infErrLogged = true; console.error('Inference error:', e); }
              if (player.currentAction) {
                player.applyActions(player.currentAction, this.humanoidData);
              }
            }
          }
        }
      }

      this.pxScene.simulate(DT);
      this.pxScene.fetchResults(true);
    }

    // Check for resets
    for (const pid of playerIds) {
      const player = this.players[pid];
      if (player && player.needsReset()) {
        this._resetPlayer(pid);
      }
    }

    // Broadcast state at STATE_HZ
    this.tickCounter++;
    if (this.tickCounter % Math.round(TICK_HZ / STATE_HZ) === 0) {
      this._broadcastState();
    }
  }

  _resetPlayer(pid) {
    const player = this.players[pid];
    if (!player) return;
    const spawnIndex = player._spawnIndex;
    const name = player.name;
    const color = player.color;

    player.destroy(this.pxScene);

    const newPlayer = new PlayerHumanoid(pid, this.px, this.physics, this.pxScene, this.material, this.humanoidData, spawnIndex);
    newPlayer.name = name;
    newPlayer.color = color;
    newPlayer._spawnIndex = spawnIndex;
    // Preserve input state
    newPlayer.moveDir = player.moveDir;
    newPlayer.faceDir = player.faceDir;
    newPlayer.speed = player.speed;
    this.players[pid] = newPlayer;
  }

  _broadcastState() {
    const state = {};
    for (const pid in this.players) {
      const player = this.players[pid];
      state[pid] = {
        name: player.name,
        color: player.color,
        poses: Array.from(player.getLinkPoses()),
      };
    }
    const msg = JSON.stringify({ type: 'state', players: state });
    this.room.broadcast(msg);
  }

  onConnect(conn, ctx) {
    console.log('Connected:', conn.id);

    if (!this.humanoidData) {
      conn.send(JSON.stringify({ type: 'waiting', message: 'Server initializing...' }));
      return;
    }

    // Clean up stale players from connections that no longer exist
    const activeIds = new Set([...this.room.getConnections()].map(c => c.id));
    for (const pid in this.players) {
      if (!activeIds.has(pid)) {
        console.log('Cleaning stale player:', pid);
        this.players[pid].destroy(this.pxScene);
        delete this.players[pid];
      }
    }

    const spawnIndex = this.globalPlayerCount++;
    const player = new PlayerHumanoid(conn.id, this.px, this.physics, this.pxScene, this.material, this.humanoidData, spawnIndex);
    player._spawnIndex = spawnIndex;
    this.players[conn.id] = player;

    // Send init data: humanoid geometry for rendering, and the body list
    conn.send(JSON.stringify({
      type: 'init',
      bodies: this.humanoidData.bodies.map(b => ({
        name: b.name,
        geoms: b.geoms,
      })),
      playerId: conn.id,
    }));

    // Notify all clients of player list
    this._broadcastPlayerList();
  }

  async onMessage(message, sender) {
    let data;
    try { data = JSON.parse(/** @type {string} */(message)); } catch(e) { return; }

    const player = this.players[sender.id];

    if (data.type === 'input' && player) {
      if (data.moveDir) player.moveDir = data.moveDir;
      if (data.faceDir) player.faceDir = data.faceDir;
      if (data.speed !== undefined) player.speed = data.speed;
      player.actionSlash = !!data.slash;
      player.actionKick = !!data.kick;
      player.actionBlock = !!data.block;
      player.actionJump = !!data.jump;
    }

    if (data.type === 'name' && player) {
      player.name = (data.name || 'Player').substring(0, 20);
      this._broadcastPlayerList();
    }
  }

  _broadcastPlayerList() {
    const list = {};
    for (const pid in this.players) {
      list[pid] = { name: this.players[pid].name, color: this.players[pid].color };
    }
    this.room.broadcast(JSON.stringify({ type: 'players', players: list }));
  }

  onClose(conn) {
    console.log('Disconnected:', conn.id);
    const player = this.players[conn.id];
    if (player) {
      player.destroy(this.pxScene);
      delete this.players[conn.id];
    }
    this._broadcastPlayerList();
  }
}

export default PartyServer;
