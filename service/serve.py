import yaml
import logging
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse
from src.serving.inference import RetrievalPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@serve.deployment(num_replicas=1, ray_actor_options={'num_cpus': 2})
class GraphRetrieverDeployment:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.pipeline = RetrievalPipeline(self.config)

    async def __call__(self, request: Request) -> JSONResponse:
        path = request.url.path

        if path == '/health':
            return JSONResponse({
                'status': 'healthy',
                'model': 'GATv2Conv-GraphRanker',
                'index_size': self.pipeline.faiss_index.ntotal,
            })

        if path == '/retrieve':
            try:
                body = await request.json()
            except Exception:
                return JSONResponse(
                    {'error': 'Invalid JSON body'}, status_code=400
                )

            query = body.get('query')
            if not query:
                return JSONResponse(
                    {'error': 'Missing required field: query'}, status_code=400
                )

            top_k = body.get('top_k', self.config['serving']['top_k'])

            result = self.pipeline.retrieve(query, top_k=top_k)
            return JSONResponse(result)

        return JSONResponse({'error': 'Not found'}, status_code=404)

def build_app(config_path: str = 'configs/serving.yaml'):
    return GraphRetrieverDeployment.bind(config_path)

app = build_app()

if __name__ == '__main__':
    serve.start(http_options={'host': '0.0.0.0', 'port': 8000})
    serve.run(app, route_prefix='/')
    import signal
    signal.pause()
