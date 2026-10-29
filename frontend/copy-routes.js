import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const distDir = path.resolve(__dirname, 'dist');
const publicDir = path.resolve(__dirname, 'public');
const indexPath = path.resolve(distDir, 'index.html');

if (!fs.existsSync(indexPath)) {
  console.error('dist/index.html not found, cannot copy routes.');
  process.exit(0);
}

const indexContent = fs.readFileSync(indexPath, 'utf-8');

// 1. Generate 404.html for Render Static Sites CDN fallback
fs.writeFileSync(path.resolve(distDir, '404.html'), indexContent, 'utf-8');
fs.writeFileSync(path.resolve(publicDir, '404.html'), indexContent, 'utf-8');
console.log('✓ Generated 404.html in dist and public');

// 2. Persona IDs to pre-generate physical HTML routes for
const knownProfiles = [
  'profile_1',
  'profile_2',
  'profile_3',
  'profile_diya_2b8989',
  'profile_riyakapoor_1aab25',
];

// Read any other persona configs from backend/app/personas/configs if available
const configsDir = path.resolve(__dirname, '..', 'backend', 'app', 'personas', 'configs');
if (fs.existsSync(configsDir)) {
  try {
    const files = fs.readdirSync(configsDir);
    for (const f of files) {
      if (f.endsWith('.json') && f !== 'persona_template.json') {
        const id = f.replace('.json', '');
        if (!knownProfiles.includes(id)) {
          knownProfiles.push(id);
        }
      }
    }
  } catch (e) {
    // ignore
  }
}

// 3. Pre-generate physical directories with index.html for known routes
const routes = [
  'settings',
  'about',
  'privacy',
  ...knownProfiles.map((p) => `chat/${p}`),
];

for (const r of routes) {
  const targetDir = path.resolve(distDir, r);
  fs.mkdirSync(targetDir, { recursive: true });
  fs.writeFileSync(path.resolve(targetDir, 'index.html'), indexContent, 'utf-8');

  const pubTargetDir = path.resolve(publicDir, r);
  fs.mkdirSync(pubTargetDir, { recursive: true });
  fs.writeFileSync(path.resolve(pubTargetDir, 'index.html'), indexContent, 'utf-8');
}

console.log(`✓ Generated physical SPA directories for ${routes.length} routes.`);
