import nest_asyncio
import asyncio
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
# from langchain_community.tools.playwright.utils import create_sync_playwright_browser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

nest_asyncio.apply()

# Entry point of the script
async def main():
    # Create an async browser
    async_browser = create_async_playwright_browser()
    
    # Create the toolkit from the browser
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    tools = toolkit.get_tools()
    # print("Tools:", tools)

    # Get tools by name
    # tools_by_name = {tool.name: tool for tool in tools}
    # navigate_tool = tools_by_name["navigate_browser"]
    # get_elements_tool = tools_by_name["get_elements"]

    # Navigate to the archived CNN World page
    # print(await navigate_tool.arun({
    #     "url": "https://python.langchain.com/"
    # })
    # )

    # print(await get_elements_tool.arun(
    # {"selector": ".container__headline", "attributes": ["innerText"]}
    # ))


    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite-preview-06-17", temperature=0
    )  # or any other LLM, e.g., ChatOpenAI(), OpenAI()

    agent_chain = create_react_agent(model=llm, tools=tools)

    result = await agent_chain.ainvoke(
        {"messages": [("user", "What is wikipedia?")]}
    )
    print(result.get("messages", [])[-1].content)




# Run the async function
if __name__ == "__main__":
    asyncio.run(main())
    # main()