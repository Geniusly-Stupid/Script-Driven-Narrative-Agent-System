from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    app_name: str = 'Script-Driven Narrative Agent System'
    mongodb_uri: str = 'mongodb://localhost:27017'
    mongodb_db: str = 'narrative_agent'
    chroma_path: str = './.chroma'
    openai_api_key: str = ''
    openai_model: str = 'gpt-4o-mini'


settings = Settings()
