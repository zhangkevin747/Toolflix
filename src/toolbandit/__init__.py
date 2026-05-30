"""ToolBandit: a clean, readable rewrite of the experiment runtime.

The whole project answers one question: can a system learn which tools
actually work by watching them succeed or fail, instead of trusting their
descriptions?

The experiment is one loop. For each task:

  1. retrieve   -> find candidate tools by text match        (retriever.py)
  2. rerank     -> the bandit orders them by learned quality (bandit.py)
  3. pick       -> choose one tool from the top slate        (loop.py / caller.py)
  4. execute    -> run it, faking failures for broken tools  (marketplace.py)
  5. reward     -> did the output help?                       (reward.py)
  6. update     -> teach the bandit from that one result      (bandit.py)

`loop.py` is that loop. Everything else is a piece it calls.
Start reading at loop.py, then bandit.py.
"""
