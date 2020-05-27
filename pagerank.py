import os
import random
import re
import sys
from random import random, choice

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    if len(corpus[page]) > 0:
        page_prob = {}
        link_prob = DAMPING / len(corpus[page])
        random_prob = (1 -  DAMPING) / len(corpus)
        total_prob = link_prob + random_prob
        for site in corpus[page]:
            page_prob.update({site : total_prob})
        for site in corpus:
            if site not in corpus[page]:
                page_prob.update({site : random_prob})
        return page_prob
    
    elif len(corpus[page]) == 0:
        page_prob = {}
        random_prob = 1 / len(corpus)
        for site in corpus:
            page_prob.update({site : random_prob})
        return page_prob

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Get list of pages, choose one randomly
    pages = []
    for site in corpus:
        pages.append(site)
    page1 = choice(pages)

    # Use transition_model to get initial weights
    next_prob = transition_model(corpus, page1, damping_factor)

    # Put weights on range 0-1 so they have distinct values 
    x = 0
    for site in next_prob:
        next_prob[site] += x
        next_prob[site] = round(next_prob[site], 4)
        x = next_prob[site]
    
    list1 = zip(next_prob.keys(), next_prob.values())
    next_prob1 = list(list1)
    site_weight = sorted(next_prob1, key=lambda x: x[1])

    # Create dict for site : count
    ranks = {}
    for site in next_prob:
        ranks.update({site : 0})

    # Generate random number (0-1)
    samples = []
    hits = []

    count = 0
    while count < SAMPLES:
        sample = random()
        samples.append(sample)
        for i in range(len(site_weight)):
            if site_weight[i][1] > sample:
                if i == 0:
                    hit = site_weight[i][0]
                    ranks[hit] += 1
                    hits.append(hit)

                    next_site = transition_model(corpus, hit, DAMPING)
                    x = 0
                    for site in next_site:
                        next_site[site] += x
                        next_site[site] = round(next_site[site], 4)
                        x = next_site[site]
        
                        list1 = zip(next_site.keys(), next_site.values())
                        next_prob1 = list(list1)
                        site_weight = sorted(next_prob1, key=lambda x: x[1])
                    count += 1
                
                else:
                    if sample > site_weight[i - 1][1]:
                        hit = site_weight[i][0]
                        ranks[hit] += 1
                        hits.append(hit)

                        next_site = transition_model(corpus, hit, DAMPING)
                        x = 0
                        for site in next_site:
                            next_site[site] += x
                            next_site[site] = round(next_site[site], 4)
                            x = next_site[site]
                        
                        list1 = zip(next_site.keys(), next_site.values())
                        next_prob1 = list(list1)
                        site_weight = sorted(next_prob1, key=lambda x: x[1])

                        count += 1

    # Turn count into probability
    for site in ranks:
        count_to_prob = ranks[site] / SAMPLES
        ranks[site] = count_to_prob
    return ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = []
    num_links = {}
    for site in corpus:
        pages.append(site)
        links = len(corpus[site])
        num_links[site] = links
    
    x = 1 / len(pages)
    page_rank = {}
    for i in range(len(pages)):
        page_rank.update({pages[i] : x})

    links_to = []    
    sum_links = {}
    change = []
    for i in range(len(pages)):
        sum_links.update({pages[i] : 0})
        change.append(0)

    rank_change = 1
    while rank_change > .001:
        for i in range(len(pages)):
            for site in corpus:
                if pages[i] in corpus[site]:
                    sum_links[pages[i]] += (page_rank[site] / num_links[site])
        for i in range(len(pages)):
            sum_links[pages[i]] = DAMPING * sum_links[pages[i]]
            sum_links[pages[i]] = (1 - DAMPING) / len(pages) + sum_links[pages[i]]   

        for i in range(len(pages)):
            change[i] = abs(page_rank[pages[i]] - sum_links[pages[i]])
            rank_change = max(change)
            page_rank[pages[i]] = sum_links[pages[i]]
            sum_links[pages[i]] = 0
    
    return page_rank



if __name__ == "__main__":
    main()
