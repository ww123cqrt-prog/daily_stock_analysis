"""Serve the existing local MLflow UI under DSA with a bilingual overlay."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import AsyncIterator, Iterable
from urllib.parse import SplitResult, urlsplit, urlunsplit

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse

from src.auth import COOKIE_NAME
from src.services.mlflow_service import start_local_mlflow_service


_RUNTIME_PATH = "/experiments/__dsa_i18n__/runtime.js"
_INJECTION_MARKER = "<!-- dsa-mlflow-i18n-runtime -->"
_CATALOG_PATH = (
    Path(__file__).resolve().parent.parent
    / "assets"
    / "mlflow_i18n"
    / "mlflow-3.14.0-zh-CN.json"
)
_ALLOWED_UPSTREAM_PORTS = frozenset({6006})
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)
_PROXY_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
_MANUAL_TRANSLATIONS = {
    "Home": "首页",
    "Experiments": "实验",
    "Runs": "运行记录",
    "Settings": "设置",
    "Welcome to MLflow": "欢迎使用 MLflow",
    "Information about UI telemetry": "关于界面使用情况统计",
    "MLflow collects usage data to improve the product. To confirm your preferences, please visit the settings page in the navigation sidebar. To learn more about what data is collected, please visit the Docs.": "MLflow 会收集匿名使用数据以改进产品。你可以在设置页确认偏好，并在文档中查看收集范围。",
    "Getting Started": "开始使用",
    "New to MLflow?": "初次使用 MLflow？",
    "Artifacts": "产物",
    "Review": "审查",
    "Tag": "标签",
}


def _load_catalog() -> dict[str, str]:
    try:
        payload = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(_MANUAL_TRANSLATIONS)

    translations = payload.get("translations") if isinstance(payload, dict) else None
    if not isinstance(translations, dict):
        return dict(_MANUAL_TRANSLATIONS)

    catalog = {
        str(source): str(translated)
        for source, translated in translations.items()
        if isinstance(source, str)
        and isinstance(translated, str)
        and source.strip()
        and translated.strip()
        and source != translated
    }
    catalog.update(_MANUAL_TRANSLATIONS)
    return catalog


def build_runtime_script(catalog: dict[str, str]) -> str:
    """Build the browser-side exact-text translator used by the MLflow UI."""

    encoded_catalog = json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))
    encoded_catalog = encoded_catalog.replace("</", "<\\/")
    return f"""(() => {{
  'use strict';
  const CATALOG = {encoded_catalog};
  const REVERSE = new Map(Object.entries(CATALOG).map(([source, translated]) => [translated, source]));
  const DYNAMIC_TRANSLATIONS = [
    [/^Metrics \\((\\d+)\\)$/, '指标 ($1)'],
    [/^Parameters \\((\\d+)\\)$/, '参数 ($1)'],
    [/^(\\d+) matching runs?$/, '$1 条匹配运行'],
    [/^(\\d+) \/ page$/, '每页 $1 条'],
  ];
  const MLFLOW_LANGUAGE_KEY = 'mlflow-ui-language';
  const DSA_LANGUAGE_KEY = 'dsa.uiLanguage';
  const TOOLBAR_ID = 'dsa-mlflow-language-toolbar';
  const ATTRIBUTES = ['placeholder', 'title', 'aria-label'];
  const textSources = new WeakMap();
  const attributeSources = new WeakMap();
  let language = readInitialLanguage();
  let scheduled = false;

  const BUSINESS_DATA_SELECTOR = [
    '[role="cell"]',
    '[role="gridcell"]',
    '[data-testid="artifact-tree-node"]',
    '[data-testid="experiment-list-item-link"]',
    '[data-testid="experiment-runs-link"]',
    '[data-testid="experiment-link"]',
  ].join(',');

  function normalizeLanguage(value) {{
    return value === 'en' || value === 'English' ? 'en' : value === 'zh' || value === 'zh-CN' ? 'zh' : null;
  }}

  function readInitialLanguage() {{
    try {{
      return normalizeLanguage(localStorage.getItem(DSA_LANGUAGE_KEY))
        || normalizeLanguage(localStorage.getItem(MLFLOW_LANGUAGE_KEY))
        || 'zh';
    }} catch (_error) {{
      return 'zh';
    }}
  }}

  function isBusinessData(element) {{
    if (!element) return false;
    if (element.closest('button, [role="button"], [data-component-type="button"]')) return false;
    if (element.closest(BUSINESS_DATA_SELECTOR)) return true;

    const keyValue = element.closest('[data-testid^="key-value-"]');
    if (keyValue && !/^key-value-Status$/i.test(keyValue.getAttribute('data-testid') || '')) {{
      return true;
    }}

    const link = element.closest('a[href]');
    if (!link) return false;
    const href = link.getAttribute('href') || '';
    if (/^#\/experiments\/[^/?]+\/runs\/[^/?]+/.test(href)) return true;
    return href === '#/experiments'
      && !link.hasAttribute('data-testid')
      && link.querySelectorAll('[role="img"]').length >= 2;
  }}

  function isIgnored(node) {{
    const element = node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement;
    return !element
      || isBusinessData(element)
      || Boolean(element.closest(`#${{TOOLBAR_ID}}, script, style, code, pre, [data-dsa-i18n-ignore]`));
  }}

  function translatedValue(source) {{
    if (CATALOG[source]) return CATALOG[source];
    for (const [pattern, replacement] of DYNAMIC_TRANSLATIONS) {{
      if (pattern.test(source)) return source.replace(pattern, replacement);
    }}
    return null;
  }}

  function translateTextNode(node) {{
    if (isIgnored(node)) return;
    const current = node.nodeValue || '';
    const trimmed = current.trim();
    if (!trimmed) return;
    const leading = current.slice(0, current.indexOf(trimmed));
    const trailing = current.slice(current.indexOf(trimmed) + trimmed.length);
    const remembered = textSources.get(node);

    if (language === 'zh') {{
      if (remembered && current === `${{leading}}${{translatedValue(remembered) || remembered}}${{trailing}}`) return;
      const translated = translatedValue(trimmed);
      if (!translated) {{
        if (remembered && current !== translatedValue(remembered)) textSources.delete(node);
        return;
      }}
      textSources.set(node, trimmed);
      node.nodeValue = `${{leading}}${{translated}}${{trailing}}`;
      return;
    }}

    if (remembered) {{
      if (trimmed === translatedValue(remembered)) node.nodeValue = `${{leading}}${{remembered}}${{trailing}}`;
      textSources.delete(node);
      return;
    }}
    const source = REVERSE.get(trimmed);
    if (source) node.nodeValue = `${{leading}}${{source}}${{trailing}}`;
  }}

  function translateAttributes(element) {{
    if (isIgnored(element)) return;
    let remembered = attributeSources.get(element);
    if (!remembered) {{
      remembered = new Map();
      attributeSources.set(element, remembered);
    }}
    for (const attribute of ATTRIBUTES) {{
      const current = element.getAttribute(attribute);
      if (!current) continue;
      const source = remembered.get(attribute);
      if (language === 'zh') {{
        if (source && current === translatedValue(source)) continue;
        const translated = translatedValue(current);
        if (translated) {{
          remembered.set(attribute, current);
          element.setAttribute(attribute, translated);
        }} else if (source && current !== translatedValue(source)) {{
          remembered.delete(attribute);
        }}
      }} else if (source) {{
        if (current === translatedValue(source)) element.setAttribute(attribute, source);
        remembered.delete(attribute);
      }} else {{
        const original = REVERSE.get(current);
        if (original) element.setAttribute(attribute, original);
      }}
    }}
  }}

  function applyTranslations(root = document.body) {{
    if (!root) return;
    enforceRouteScope();
    applyScope(root);
    if (root.nodeType === Node.TEXT_NODE) translateTextNode(root);
    if (root.nodeType === Node.ELEMENT_NODE) translateAttributes(root);
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
    let node = walker.nextNode();
    while (node) {{
      if (node.nodeType === Node.TEXT_NODE) translateTextNode(node);
      else translateAttributes(node);
      node = walker.nextNode();
    }}
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    updateToolbar();
  }}

  function scheduleTranslations() {{
    if (scheduled) return;
    scheduled = true;
    window.requestAnimationFrame(() => {{
      scheduled = false;
      applyTranslations();
    }});
  }}

  function captureScrollPositions() {{
    const elements = Array.from(document.querySelectorAll('*'))
      .filter((element) => element.scrollTop || element.scrollLeft)
      .map((element) => ({{ element, top: element.scrollTop, left: element.scrollLeft }}));
    return {{ x: window.scrollX, y: window.scrollY, elements }};
  }}

  function restoreScrollPositions(snapshot) {{
    window.scrollTo(snapshot.x, snapshot.y);
    snapshot.elements.forEach((item) => item.element.scrollTo(item.left, item.top));
  }}

  function persistLanguage(nextLanguage) {{
    const scrollSnapshot = captureScrollPositions();
    language = nextLanguage;
    try {{
      localStorage.setItem(MLFLOW_LANGUAGE_KEY, nextLanguage);
      localStorage.setItem(DSA_LANGUAGE_KEY, nextLanguage);
    }} catch (_error) {{}}
    applyTranslations();
    restoreScrollPositions(scrollSnapshot);
    window.requestAnimationFrame(() => restoreScrollPositions(scrollSnapshot));
  }}

  function experimentRunsHash(experimentId) {{
    return '#/experiments/' + experimentId + '/runs';
  }}

  function scopedHash(hash) {{
    if (!hash || hash === '#' || hash === '#/') return '#/experiments';
    if (/^#\/experiments(?:\?|$)/.test(hash)) return hash;
    if (/^#\/compare-runs(?:[/?]|$)/.test(hash)) return hash;
    if (/^#\/experiments\/[^/?]+\/runs(?:[/?]|$)/.test(hash)) return hash;
    const experimentRoute = hash.match(/^#\/experiments\/([^/?]+)/);
    if (experimentRoute) return experimentRunsHash(experimentRoute[1]);
    return '#/experiments';
  }}

  function enforceRouteScope() {{
    const nextHash = scopedHash(window.location.hash);
    if (nextHash !== window.location.hash) window.location.replace(nextHash);
  }}

  function applyScope(root) {{
    if (!root.querySelectorAll) return;
    root.querySelectorAll('a[href]').forEach((link) => {{
      const href = link.getAttribute('href') || '';
      if (/mlflow\.org\/docs\/latest\/genai\//.test(href)) {{
        (link.closest('li') || link).setAttribute('data-dsa-scope-hidden', 'true');
        return;
      }}
      const experiment = href.match(/^#\/experiments\/([^/?]+)(?:\?.*)?$/);
      if (experiment) link.setAttribute('href', experimentRunsHash(experiment[1]));
      const currentHref = link.getAttribute('href') || '';
      if (currentHref.startsWith('#') && scopedHash(currentHref) !== currentHref) {{
        (link.closest('li') || link).setAttribute('data-dsa-scope-hidden', 'true');
      }}
    }});
    root.querySelectorAll('nav *, aside *').forEach((element) => {{
      if (!element.children.length && element.textContent.trim() === 'GenAI') {{
        element.setAttribute('data-dsa-scope-hidden', 'true');
      }}
    }});
    root.querySelectorAll('nav [role="button"], aside [role="button"]').forEach((element) => {{
      const text = element.textContent.trim();
      if (text === 'GenAI' || text.startsWith('Assistant') || text.startsWith('助手')) {{
        element.setAttribute('data-dsa-scope-hidden', 'true');
      }}
    }});
  }}

  function createToolbar() {{
    if (document.getElementById(TOOLBAR_ID)) return;
    const toolbar = document.createElement('div');
    toolbar.id = TOOLBAR_ID;
    toolbar.setAttribute('data-dsa-i18n-ignore', 'true');
    toolbar.innerHTML = `
      <a href="/research" title="返回 DSA 研究页">DSA</a>
      <span class="dsa-divider"></span>
      <a href="/research" data-dsa-nav="research"></a>
      <a href="/workflow" data-dsa-nav="workflow"></a>
      <a href="/backtest" data-dsa-nav="backtest"></a>
      <span class="dsa-divider"></span>
      <button type="button" data-language="zh">中文</button>
      <span class="dsa-separator">|</span>
      <button type="button" data-language="en">English</button>`;
    const style = document.createElement('style');
    style.textContent = `
      #${{TOOLBAR_ID}} {{ position: fixed; top: 10px; right: 18px; z-index: 2147483647;
        display: flex; align-items: center; flex-wrap: wrap; gap: 7px; min-height: 32px; max-width: calc(100vw - 36px); padding: 0 10px;
        border: 1px solid rgba(107,114,128,.35); border-radius: 6px;
        background: rgba(255,255,255,.96); color: #374151; font: 12px/1.2 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        box-shadow: 0 2px 8px rgba(0,0,0,.12); }}
      #${{TOOLBAR_ID}} a, #${{TOOLBAR_ID}} button {{ color: inherit; text-decoration: none; background: none;
        border: 0; padding: 3px 1px; font: inherit; cursor: pointer; }}
      #${{TOOLBAR_ID}} a:hover, #${{TOOLBAR_ID}} button:hover {{ color: #0b57d0; }}
      #${{TOOLBAR_ID}} button[aria-pressed="true"] {{ color: #0b57d0; font-weight: 600; }}
      #${{TOOLBAR_ID}} .dsa-divider {{ width: 1px; height: 14px; background: rgba(107,114,128,.35); }}
      #${{TOOLBAR_ID}} .dsa-separator {{ color: #9ca3af; }}
      [data-dsa-scope-hidden="true"] {{ display: none !important; }}
      @media (prefers-color-scheme: dark) {{ #${{TOOLBAR_ID}} {{ background: rgba(31,41,55,.96); color: #e5e7eb; }} }}
      @media (max-width: 640px) {{
        aside {{ display: none !important; }}
        #${{TOOLBAR_ID}} {{ top: auto; right: 8px; bottom: 8px; max-width: calc(100vw - 16px); gap: 5px; }}
      }}
    `;
    document.head.appendChild(style);
    document.body.appendChild(toolbar);
    toolbar.querySelectorAll('button[data-language]').forEach((button) => {{
      button.addEventListener('click', () => persistLanguage(button.dataset.language));
    }});
  }}

  function updateToolbar() {{
    const toolbar = document.getElementById(TOOLBAR_ID);
    if (!toolbar) return;
    const labels = language === 'zh'
      ? {{ research: '研究', workflow: '工作流', backtest: '回测' }}
      : {{ research: 'Research', workflow: 'Workflow', backtest: 'Backtest' }};
    toolbar.querySelectorAll('[data-dsa-nav]').forEach((link) => {{
      link.textContent = labels[link.dataset.dsaNav] || '';
    }});
    toolbar.querySelectorAll('button[data-language]').forEach((button) => {{
      button.setAttribute('aria-pressed', String(button.dataset.language === language));
    }});
  }}

  function start() {{
    enforceRouteScope();
    createToolbar();
    applyTranslations();
    new MutationObserver(scheduleTranslations).observe(document.body, {{
      childList: true,
      subtree: true,
      characterData: true,
      attributes: true,
      attributeFilter: ATTRIBUTES,
    }});
    window.addEventListener('storage', (event) => {{
      if (event.key !== DSA_LANGUAGE_KEY && event.key !== MLFLOW_LANGUAGE_KEY) return;
      const nextLanguage = normalizeLanguage(event.newValue);
      if (nextLanguage && nextLanguage !== language) {{
        language = nextLanguage;
        applyTranslations();
      }}
    }});
    window.addEventListener('hashchange', enforceRouteScope);
  }}

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, {{ once: true }});
  else start();
}})();"""


def inject_runtime_script(raw: bytes) -> bytes:
    """Inject the external runtime once while leaving the upstream bundle intact."""

    try:
        html = raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw
    if _INJECTION_MARKER in html:
        return raw
    snippet = (
        f'{_INJECTION_MARKER}<script defer src="{_RUNTIME_PATH}"></script>'
    )
    closing_body = re.search(r"</body\s*>", html, flags=re.IGNORECASE)
    if closing_body:
        html = html[: closing_body.start()] + snippet + html[closing_body.start() :]
    else:
        html += snippet
    return html.encode("utf-8")


def _validate_upstream_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in _LOOPBACK_HOSTS
        or parsed.port not in _ALLOWED_UPSTREAM_PORTS
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("MLflow UI upstream must be the local http://127.0.0.1:6006 service")
    host = f"[{parsed.hostname}]" if ":" in (parsed.hostname or "") else parsed.hostname
    return f"http://{host}:{parsed.port}/"


def _request_headers(request: Request) -> list[tuple[str, str]]:
    headers: list[tuple[str, str]] = []
    for key, value in request.headers.items():
        lowered = key.lower()
        if lowered in _HOP_BY_HOP_HEADERS or lowered in {"host", "accept-encoding"}:
            continue
        if lowered == "cookie":
            value = "; ".join(
                part.strip()
                for part in value.split(";")
                if part.strip() and not part.strip().startswith(f"{COOKIE_NAME}=")
            )
            if not value:
                continue
        headers.append((key, value))
    headers.append(("accept-encoding", "identity"))
    return headers


def _response_headers(headers: httpx.Headers, *, body_rewritten: bool) -> list[tuple[bytes, bytes]]:
    forwarded: list[tuple[bytes, bytes]] = []
    for key, value in headers.multi_items():
        lowered = key.lower()
        if lowered in _HOP_BY_HOP_HEADERS:
            continue
        if lowered in {"date", "server"}:
            continue
        if body_rewritten and lowered in {
            "cache-control",
            "content-length",
            "content-encoding",
            "etag",
            "last-modified",
        }:
            continue
        forwarded.append((key.encode("latin-1"), value.encode("latin-1")))
    return forwarded


def _rewrite_location(location: str, upstream: SplitResult) -> str:
    parsed = urlsplit(location)
    if parsed.scheme or parsed.netloc:
        if parsed.hostname != upstream.hostname or parsed.port != upstream.port:
            return location
        path = parsed.path
    else:
        path = parsed.path
    normalized_path = "/" + path.lstrip("/")
    proxied_path = "/experiments/" if normalized_path == "/" else f"/experiments{normalized_path}"
    return urlunsplit(("", "", proxied_path, parsed.query, parsed.fragment))


def _set_response_headers(
    response: Response,
    headers: Iterable[tuple[bytes, bytes]],
    *,
    upstream: SplitResult,
) -> None:
    rewritten: list[tuple[bytes, bytes]] = []
    for key, value in headers:
        if key.lower() == b"location":
            value = _rewrite_location(value.decode("latin-1"), upstream).encode("latin-1")
        elif key.lower() == b"set-cookie":
            cookie = value.decode("latin-1")
            if cookie.split("=", 1)[0].strip() == COOKIE_NAME:
                continue
            cookie = re.sub(
                r"(;\s*path=)/(?=;|$)",
                r"\1/experiments/",
                cookie,
                flags=re.IGNORECASE,
            )
            value = cookie.encode("latin-1")
        rewritten.append((key, value))
    response.raw_headers = rewritten


def _upstream_path(path: str) -> str:
    segments = path.split("/") if path else []
    if any(segment in {".", ".."} for segment in segments):
        raise ValueError("Invalid MLflow proxy path")
    return path


def create_mlflow_ui_router(
    upstream_url: str = "http://127.0.0.1:6006/",
    transport: httpx.AsyncBaseTransport | None = None,
) -> APIRouter:
    """Create the router mounted by DSA at ``/experiments``."""

    upstream_base = _validate_upstream_url(upstream_url)
    upstream = urlsplit(upstream_base)
    router = APIRouter()
    catalog = _load_catalog()
    runtime_script = build_runtime_script(catalog).encode("utf-8")

    @router.get("", include_in_schema=False)
    async def redirect_to_slash() -> RedirectResponse:
        return RedirectResponse(url="/experiments/", status_code=307)

    @router.get("/__dsa_i18n__/runtime.js", include_in_schema=False)
    async def runtime() -> Response:
        return Response(
            runtime_script,
            media_type="text/javascript; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    @router.post("/__dsa_control__/start", include_in_schema=False)
    async def start_mlflow() -> JSONResponse:
        try:
            result = start_local_mlflow_service()
        except (OSError, RuntimeError):
            return JSONResponse(
                {"status": "error", "message": "MLflow 启动失败，请检查本机日志。"},
                status_code=500,
            )
        status_code = 200 if result.get("status") == "available" else 202
        return JSONResponse(result, status_code=status_code)

    async def proxy_request(request: Request, path: str = "") -> Response:
        try:
            safe_path = _upstream_path(path)
        except ValueError as exc:
            return Response(str(exc), status_code=400, media_type="text/plain")

        target = upstream_base + safe_path
        if request.url.query:
            target += "?" + request.url.query

        client = httpx.AsyncClient(transport=transport, follow_redirects=False, timeout=60.0)
        upstream_request = client.build_request(
            request.method,
            target,
            headers=_request_headers(request),
            content=request.stream(),
        )
        try:
            upstream_response = await client.send(upstream_request, stream=True)
        except httpx.RequestError:
            await client.aclose()
            return HTMLResponse(
                """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>MLflow 服务未启动</title></head>
<body><main style="max-width:640px;margin:80px auto;font:16px/1.6 sans-serif">
<h1>MLflow 服务未启动</h1><p>DSA 无法连接本机 6006 端口。可以启动固定的本机 MLflow 服务，或重新检查已有服务。</p>
<p id="status" role="status"></p><button id="start" type="button">启动 MLflow</button>
<button type="button" onclick="location.reload()">重新检查</button>
<script>document.getElementById('start').addEventListener('click',async function(){this.disabled=true;const status=document.getElementById('status');status.textContent='正在启动 MLflow...';try{const response=await fetch('/experiments/__dsa_control__/start',{method:'POST',credentials:'same-origin'});const payload=await response.json();if(!response.ok)throw new Error(payload.message||'启动失败');status.textContent=payload.status==='available'?'MLflow 已启动，正在重新检查...':'MLflow 正在启动，稍后自动重新检查...';setTimeout(()=>location.reload(),1200);}catch(error){status.textContent=error.message||'MLflow 启动失败';this.disabled=false;}});</script>
</main></body></html>""",
                status_code=502,
                headers={"Cache-Control": "no-store"},
            )

        content_type = upstream_response.headers.get("content-type", "").lower()
        is_html = request.method != "HEAD" and "text/html" in content_type
        if is_html:
            try:
                body = await upstream_response.aread()
            finally:
                await upstream_response.aclose()
                await client.aclose()
            body = inject_runtime_script(body)
            response = Response(body, status_code=upstream_response.status_code)
            headers = _response_headers(upstream_response.headers, body_rewritten=True)
            headers.append((b"cache-control", b"no-store"))
            headers.append((b"content-length", str(len(body)).encode("ascii")))
            _set_response_headers(response, headers, upstream=upstream)
            return response

        if request.method == "HEAD":
            await upstream_response.aclose()
            await client.aclose()
            response = Response(status_code=upstream_response.status_code)
            _set_response_headers(
                response,
                _response_headers(upstream_response.headers, body_rewritten=False),
                upstream=upstream,
            )
            return response

        if upstream_response.is_stream_consumed:
            body = upstream_response.content
            await upstream_response.aclose()
            await client.aclose()
            response = Response(body, status_code=upstream_response.status_code)
            _set_response_headers(
                response,
                _response_headers(upstream_response.headers, body_rewritten=False),
                upstream=upstream,
            )
            return response

        async def stream_body() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream_response.aiter_raw():
                    yield chunk
            finally:
                await upstream_response.aclose()
                await client.aclose()

        response = StreamingResponse(stream_body(), status_code=upstream_response.status_code)
        _set_response_headers(
            response,
            _response_headers(upstream_response.headers, body_rewritten=False),
            upstream=upstream,
        )
        return response

    router.add_api_route("/", proxy_request, methods=_PROXY_METHODS, include_in_schema=False)
    router.add_api_route("/{path:path}", proxy_request, methods=_PROXY_METHODS, include_in_schema=False)
    return router
