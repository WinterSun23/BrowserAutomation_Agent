"""Save a login for the agent: opens a visible browser, you sign in yourself, press Enter, and the session (cookies)
is written to a file. Then run the agent with SITEGRAPH_STORAGE_STATE=<file> to start already signed in.
Usage: python -m sitegraph.browser.login <url> <out.json>"""
import sys

from playwright.sync_api import sync_playwright

url, out = sys.argv[1], sys.argv[2]
with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=False)
    context = browser.new_context()
    context.new_page().goto(url)
    input("Sign in in the browser window, then press Enter here... ")
    context.storage_state(path=out)
    browser.close()
print("saved", out, "- keep this file private; it is a login session")
