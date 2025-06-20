"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from http import HTTPStatus

from aiohttp import web
from botbuilder.core.integration import aiohttp_error_middleware
from botbuilder.core.invoke_response import InvokeResponse

import sys
sys.path.append("./")
sys.path.append("../")

from teamsBot import teamsApp

routes = web.RouteTableDef()

@routes.post("/api/messages")
async def on_messages(req: web.Request) -> web.Response:
    # if "application/json" in req.headers["Content-Type"]:
    #     body = await req.json()
    #     print(body)

    res = await teamsApp.process(req)

    if res is not None:
        return res

    return web.Response(status=HTTPStatus.OK)

app = web.Application(middlewares=[aiohttp_error_middleware])
app.add_routes(routes)

from config import Config

if __name__ == "__main__":
    web.run_app(app, host="localhost", port=Config.PORT)