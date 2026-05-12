# CSE 150A: Bayesian Network for Programming Skill Prediction

## Team Members
- Lianna Lim
- Iha
- Ioanna
- Ved

## Dataset

Stack Overflow Developer Survey 2023:
https://www.kaggle.com/datasets/mahdialfred/stack-overflow-developer-survey-2023?select=so_survey_2023.pdf

## Motivation and Problem Statement
The problem we are tackling in this project is to understand how the use of AI-assisted coding tools influences programming skill among developers. As AI tools become more common in software development, it is important to understand whether they actually help developers improve or if they create over-reliance.
This problem is important because programming skill is influenced by multiple factors, such as experience, practice, and tool usage. The effect of AI tools is not the same for everyone and can vary depending on how developers interact with them.
Uncertainty modeling is important here because these relationships are not fixed or predictable. Fixed relationships assume that variables are connected by exact rules (e.g., skill always increases linearly with experience), whereas in reality these relationships are uncertain and vary across individuals. For example, two developers with the same experience level may have different skill levels depending on their level of reliance on AI tools and how frequently they use them. This kind of variability makes it difficult to model the problem using simple deterministic methods.
Non-probabilistic approaches, such as basic regression models, assume fixed relationships between variables and do not capture uncertainty or conditional dependencies well. In contrast, Bayesian networks allow us to model how different factors influence each other and reason about probabilities, making them a better fit for this problem.

## PEAS Analysis
Performance measure: accuracy of determining programming skill and speed of computing output
Environment: set of developers
Actuators: display of an individual’s probabilistic programming skill
Sensors: AI Usage (AISelect), Programming Experience (YearsCodePro), Coding Practice (YearsCode), Project Activity (CodingActivities), Programming Skill (ConvertedCompYearly as a proxy)


