import argparse
import signal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mii
import requests
import uvicorn

app = FastAPI()

# 全局变量
global restful_api_port
global deployment_name


class CompletionRequest(BaseModel):
    prompts: list[str]
    temperature: float = 0.7
    max_length: int = 8192
    min_new_tokens: int = 1
    max_new_tokens: int = 8192
    ignore_eos: bool = False  # False或者None遇到EOS就停止生成，True遇到EOS也继续生成
    top_p: float = 0.7
    top_k: int = 3
    do_sample: bool = True
    return_full_text: bool = False


@app.post("/api/v1/chat/completions")
async def completion(request_data: CompletionRequest):
    url = f"http://localhost:{restful_api_port}/mii/{deployment_name}"
    response = requests.post(url, json=request_data.json())
    return response.json()


@app.post("/stop_model")
async def stop_model():
    try:
        client = mii.client()
        client.terminate_server()
        return {"message": "Model stopped successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def start_model(m_name, d_name, r_port, t_parallel, r_num, torch_dist_port: int = 29500):
    global restful_api_port
    global deployment_name
    restful_api_port = r_port
    deployment_name = d_name
    mii.serve(
        m_name,
        deployment_name=d_name,
        tensor_parallel=t_parallel,
        replica_num=r_num,
        enable_restful_api=True,
        restful_api_port=r_port,
        torch_dist_port=torch_dist_port,
    )


def stop_model_service():
    try:
        client = mii.client()
        client.terminate_server()
    except Exception as e:
        print(f"Error stopping model service: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a FastAPI server with MII model.')
    parser.add_argument('-m', '--model_name', type=str, required=True, help='Name of the model to serve.')
    parser.add_argument('-n', '--deployment_name', type=str, default='mistral-deployment',
                        help='Deployment name (default: mistral-deployment).')
    parser.add_argument('--restful_api_port', type=int, default=28080, help='RESTful API port (default: 28080).')
    parser.add_argument('-t', '--tensor_parallel', type=int, default=1, help='Tensor parallelism degree (default: 1).')
    parser.add_argument('-p', '--replica_num', type=int, default=1, help='Number of model replicas (default: 1).')
    parser.add_argument('-H', '--host', type=str, default='0.0.0.0',
                        help='Host for the FastAPI server (default: 0.0.0.0).')
    parser.add_argument('-P', '--port', type=int, default=8000, help='Port for the FastAPI server (default: 8000).')
    parser.add_argument('-d', '--torch_dist_port', type=int, default=29500, required=False,
                        help="Torch dist port (default: 29500).")

    args = parser.parse_args()


    def signal_handler(sig, frame):
        print('Stopping model service...')
        stop_model_service()
        print('Model service stopped.')
        uvicorn.main.shutdown()


    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    start_model(args.model_name, args.deployment_name, args.restful_api_port, args.tensor_parallel, args.replica_num,
                args.torch_dist_port)
    uvicorn.run(app, host=args.host, port=args.port)
