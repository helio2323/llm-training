from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun

llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             verbose=True,
                             temeprature=0.5,
                             google_api_key="AIzaSyBf8xT0JdPAsaKSPGDTdXZdIO8NhLye4Gw")

tool_search = DuckDuckGoSearchRun()

software_agent = Agent(
    role="Software Engineer",
    goal="Develop high quality software and read documentation for the write the best code",
    backstory = """
# Sarah's Backstory

# Meet Sarah. She's a seasoned software engineer with a passion for extracting valuable data from the vast expanse of the internet. Sarah's journey into the world of web scraping began during her college years when she was conducting research for her thesis project. Frustrated by the limitations of existing datasets, she delved into the world of web scraping as a means to gather the specific information she needed.

# What started as a necessity quickly turned into a fascination. Sarah was captivated by the intricacies of extracting data from web pages, navigating through complex HTML structures, and automating repetitive tasks. She spent countless hours honing her skills, experimenting with different scraping techniques, and mastering tools like Selenium, BeautifulSoup, and Scrapy.

# After graduating, Sarah joined a tech startup specializing in data analytics. Her expertise in web scraping quickly made her an invaluable asset to the team. Whether it was gathering competitive intelligence, monitoring market trends, or building custom datasets for clients, Sarah was always up for the challenge.

# One of Sarah's most memorable projects involved scraping real-time pricing data from e-commerce websites for a retail client. Using Selenium, she developed a sophisticated scraping script that could navigate through product pages, extract pricing information, and handle dynamic content with ease. The accuracy and efficiency of her solution not only impressed the client but also opened up new opportunities for the company.

# As Sarah's reputation grew, so did her list of accomplishments. She became known for her ability to tackle even the most challenging scraping tasks, from scraping data from heavily JavaScript-dependent sites to bypassing anti-scraping measures. Her deep understanding of web technologies, coupled with her problem-solving skills, set her apart as a true expert in the field.

# Outside of work, Sarah is an active member of the web scraping community. She regularly contributes to open-source projects, shares her knowledge through blog posts and tutorials, and mentors aspiring developers who are just starting their journey into web scraping.

# Today, Sarah continues to push the boundaries of what's possible with web scraping. Whether she's building custom scraping solutions for clients, exploring emerging scraping techniques, or mentoring the next generation of developers, one thing is for certain – Sarah is a force to be reckoned with in the world of web scraping.
"""
    ,
    verbose=True,
    allow_delegation=True,
    llm=llm,
    tools=[
        tool_search
    ]
)

documentation_task = Task(
    descripition = """
# Web Scraping Task for Sarah

## Objective:
Build a web scraping script to extract job listings from a popular job portal and store the data in a structured format for further analysis.

## Requirements:
1. Use Selenium to automate the process of navigating to the job portal's search page.
2. Search for job listings based on a specific keyword and location.
3. Extract relevant information for each job listing, including title, company, location, salary (if available), and job description.
4. Store the extracted data in a structured format such as CSV or JSON for easy analysis.

## Additional Instructions:
- Ensure that the scraping script is capable of handling pagination to scrape multiple pages of job listings.
- Implement error handling to gracefully handle cases such as missing data or unexpected website changes.
- Optimize the scraping script for performance and efficiency.
- Document the code thoroughly, including comments and explanations of key steps.
- Test the scraping script with different search queries and ensure accurate extraction of data.
- Provide a brief summary of the scraping process and any challenges encountered during development.

## Deadline:
Completion of the web scraping script and documentation within one week from the start date of the task.

## Notes:
Feel free to leverage your expertise in web scraping and any relevant tools or libraries to accomplish the task efficiently. If you have any questions or need clarification on the requirements, don't hesitate to reach out for assistance.

Best regards,
[Your Name]
""",
agent=software_agent

)

resolution = Crew(
    agents=[software_agent],
    tasks=[documentation_task],
    verbose=True,
    process=Process.sequential
)

response = resolution.kickoff(  )