def route_query(question):

    question = question.lower()

    cricket_keywords = [
        "cricket",
        "batsman",
        "bowler",
        "wicket",
        "odi",
        "test",
        "ipl",
        "virat",
        "dhoni",
        "sachin"
    ]

    olympic_keywords = [
        "olympic",
        "medal",
        "athletics",
        "javelin",
        "wrestling",
        "boxing",
        "neeraj",
        "pv sindhu",
        "mary kom"
    ]


    cricket = any(word in question for word in cricket_keywords)
    olympics = any(word in question for word in olympic_keywords)


    if cricket and olympics:
        return "both"

    elif cricket:
        return "cricket"

    elif olympics:
        return "olympics"

    else:
        return "unknown"


while True:

    q = input("\nAsk Question : ")

    if q.lower() == "exit":
        break

    print("Route :", route_query(q))