def ideas_gen_prompt(n_ideas):
    return f"""{n_ideas}"""  # Your prompt goes here

def idea_vars_gen_prompt(idea, n_vars):
    return f"""{idea}{n_vars}"""  # Your prompt goes here

def story_gen_prompt(idea, n_steps):
    return f"""{idea}{n_steps}"""  # Your prompt goes here

def is_img_unsafe_prompt():
    return ""  # Your prompt goes here