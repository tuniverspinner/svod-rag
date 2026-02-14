def handler(request, response):
    from app.main import app
    return app(request, response)
