"""Provider clients: OpenAI (images + storyboard text), video, and audio.

Third-party SDKs are imported lazily inside each client so that --dry-run and
--help work even when credentials/SDKs are not configured.
"""
