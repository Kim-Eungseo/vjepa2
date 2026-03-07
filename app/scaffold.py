# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(app, args, resume_preempt=False):
    module = args.get("module", "train")
    logger.info(f"Running {module} of app: {app}")
    return importlib.import_module(f"app.{app}.{module}").main(args=args, resume_preempt=resume_preempt)
