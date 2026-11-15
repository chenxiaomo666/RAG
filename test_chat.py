from agent import chat


user_input = "哆啦A梦的三个密码道具是什么？"
reranked_chunks = []
res = chat(user_input, reranked_chunks)
print(res)
print(res.get("choices")[0].get("message").get("content"))
