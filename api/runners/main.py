import logging
import threading

import zmq

from api.config import MAX_CONCURRENT_TASKS

from .tasks import process_task
from api.logger import logger
from api.mcp import MCPBridgeClient

logging.getLogger("httpx").setLevel(logging.WARNING)


def worker_routine(worker_url, context, mcp_bridge):
    """Worker routine"""

    # Socket to pull messages from the dispatcher
    receiver = context.socket(zmq.PULL)
    receiver.connect(worker_url)

    while True:
        try:
            # Receive a task
            message = receiver.recv_json()

            # Process the task
            process_task(message, mcp_bridge)
        except Exception as e:
            logger.exception("An error occurred in the ZMQ main loop: %s", e)


def main(worker_url="tcp://localhost:5556", sender_url="tcp://localhost:5555"):
    """server routine"""

    # Prepare our context and sockets
    context = zmq.Context()

    # Socket to pull messages from clients
    receiver = context.socket(zmq.PULL)  # Receives messages
    receiver.bind(sender_url)

    # Socket to push messages to workers
    distributor = context.socket(zmq.PUSH)  # Distributes work
    distributor.bind(worker_url)

    # MCP Bridge client initalization
    mcp_bridge = MCPBridgeClient()

    # Launch pool of worker threads
    for i in range(MAX_CONCURRENT_TASKS):
        thread = threading.Thread(target=worker_routine, args=(worker_url, context, mcp_bridge))
        thread.start()

    logger.info(f"ZeroMQ is listening at worker:{worker_url} | receiver:{sender_url}")
    zmq.device(zmq.STREAMER, receiver, distributor)

    # Cleanup
    receiver.close()
    distributor.close()
    context.term()


if __name__ == "__main__":
    main()
