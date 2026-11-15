/* Crypto Master AI PWA: cache only public app-shell assets. Never cache API/auth/scan data. */
const CACHE_NAME = "crypto-master-ai-shell-v1";
const APP_SHELL = ["/", "/offline", "/static/app.css", "/static/manifest.webmanifest"];
self.addEventListener("install", event => {
  event.waitUntil(caches.open(CACHE_NAME).then(cache => cache.addAll(APP_SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", event => {
  event.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", event => {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== "GET" || url.origin !== self.location.origin) return;
  if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/auth/")) return;
  if (request.mode === "navigate") {
    event.respondWith(fetch(request).catch(async () => (await caches.match("/offline")) || new Response("You are offline. Reconnect to use Crypto Master AI.", {status: 503, headers: {"Content-Type":"text/plain; charset=utf-8"}})));
    return;
  }
  if (["/static/app.css", "/static/manifest.webmanifest"].includes(url.pathname)) {
    event.respondWith(caches.match(request).then(cached => cached || fetch(request).then(response => {
      if (response.ok) { const copy = response.clone(); caches.open(CACHE_NAME).then(cache => cache.put(request, copy)); }
      return response;
    })));
  }
});
