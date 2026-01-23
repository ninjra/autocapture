import React from 'react';
import { createRoot } from 'react-dom/client';
import './legacy/app.js';

const root = document.getElementById('console-root');
if (root) {
  createRoot(root).render(<></>);
}
