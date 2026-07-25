/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** MLflow 实验跟踪 UI 地址，默认 http://127.0.0.1:6006/ */
  readonly VITE_MLFLOW_URL?: string
  /** CatDesk 9点猫研服务地址，默认 http://127.0.0.1:8765/ */
  readonly VITE_CATDESK_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
