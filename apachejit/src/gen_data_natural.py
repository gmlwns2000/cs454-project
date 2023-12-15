import gc
import os
import json
from pydriller import Repository, Commit
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import datetime
# from git_summary import *
# from git_summary2 import *
from git_natural import *

proj_repo = {
    # 'AMQ': ['apache/activemq'], 
    # 'HBASE': ['apache/hbase'],
    # 'SPARK': ['apache/spark'], 
    # 'KAFKA': ['apache/kafka'], 
    # 'GROOVY': ['apache/groovy'],
    # 'ZEPPELIN': ['apache/zeppelin'], 
    # 'HDFS': ['apache/hadoop-hdfs', 'apache/hadoop'],
    # 'FLINK': ['apache/flink'], 
    
    # # 'CAMEL': ['apache/camel'], 
    
    # 'MAPREDUCE': ['apache/hadoop-mapreduce', 'apache/hadoop'], 
    # 'IGNITE': ['apache/ignite'],
    # 'CASSANDRA': ['apache/cassandra'], 
    # 'HIVE': ['apache/hive'], 
    'ZOOKEEPER': ['apache/zookeeper']
}

repo_names = [
    # 'apache/activemq', 
    # 'apache/hbase', 
    # 'apache/spark', 
    # 'apache/kafka', 
    # 'apache/groovy', 
    # 'apache/zeppelin', 
    # 'apache/hadoop-hdfs', 
    # 'apache/hadoop', 
    # 'apache/flink', 
    
    # # 'apache/camel', 
    
    # 'apache/hadoop-mapreduce', 
    # 'apache/ignite', 
    # 'apache/cassandra', 
    # 'apache/hive', 
    'apache/zookeeper'
]


# BASE_DIR = os.path.dirname(os.path.dirname(__file__))
BASE_DIR = '/home/rlatnghks17023/cs454-project/apachejit'
data_path = os.path.join(BASE_DIR, 'data')
repo_path = os.path.join(BASE_DIR, 'repos')
json_path = os.path.join(BASE_DIR, "json")

def commit_diff(commit: Commit):
    diff = ""
    for file in commit.modified_files:
        diff = diff + "\n\n" + file.diff
    return diff

def valid_commit(commit: Commit):
    for file in commit.modified_files:
        if not str.isspace(file.diff):
            return True
    return False

if not os.path.exists(os.path.join(json_path, "bughash_links.json")) or not os.path.exists(os.path.join(json_path, "hash_index.json")):
    # Organize bug hash - fix hash info
    bughash_links = dict()
    hash_index = dict()
    for project in tqdm(proj_repo.keys(), dynamic_ncols=True):
        print(project)
        commit_link_filename = os.path.join(data_path, f"commit_links_{project}.csv")
        df = pd.read_csv(commit_link_filename)
        for index, row in df.iterrows():
            bughash_links[row["bug_hash"]] = row["fix_hash"]

    fix_hashes = set(bughash_links.values())
    # Organize hash - index info
    for project in tqdm(proj_repo.keys(), dynamic_ncols=True):
        repos = [os.path.join(repo_path, r.split('/')[-1]) for r in proj_repo[project]]
        for repo in repos:
            index = 0
            for commit in tqdm(Repository(repo, since=datetime.datetime(2003, 9, 11, 14, 11, 56), to=datetime.datetime(2019, 12, 26, 18, 29, 9)).traverse_commits()):
                if not valid_commit(commit=commit) and commit.hash not in fix_hashes: continue
                hash_index[commit.hash] = index
                index += 1

    with open(os.path.join(json_path, "bughash_links.json"), "w") as outfile: 
        json.dump(bughash_links, outfile, indent=2)

    with open(os.path.join(json_path, "hash_index.json"), "w") as outfile: 
        json.dump(hash_index, outfile, indent=2)


# Organize data
def create_json(repo: str):
    with open(os.path.join(json_path, "bughash_links.json")) as json_file:
        bughash_links = json.load(json_file)
    with open(os.path.join(json_path, "hash_index.json")) as json_file:
        hash_index = json.load(json_file)
    
    fix_hashes = set(bughash_links.values())
    r = os.path.split(repo)[-1]
    data = dict()
    data[r] = []
    for commit in Repository(repo, since=datetime.datetime(2003, 9, 11, 14, 11, 56), to=datetime.datetime(2019, 12, 26, 18, 29, 9)).traverse_commits():
        commit_data = dict()
        if not valid_commit(commit=commit) and commit.hash not in fix_hashes: continue
        diff = commit_diff(commit=commit)
        refine_diff = git_summary(diff)
        commit_data["text"] = refine_diff
        if commit.hash in bughash_links.keys():
            commit_data["is_buggy"] = 1
            commit_data["index"] = hash_index[commit.hash]
            if bughash_links[commit.hash] in hash_index:
                commit_data["reported_index"] = hash_index[bughash_links[commit.hash]]
            else:
                commit_data["reported_index"] = 99999
        else:
            commit_data["is_buggy"] = 0
            commit_data["index"] = hash_index[commit.hash]
            commit_data["reported_index"] = -1
        data[r].append(commit_data)
    with open(os.path.join(json_path, f"{r}_ex.json"), "w") as outfile: 
        json.dump(data, outfile, indent=2)

if __name__=="__main__":
    p = Pool(15)
    repos = [os.path.join(repo_path, r.split('/')[-1]) for r in repo_names]
    empty = []
    for d in tqdm(p.imap_unordered(create_json, repos)):
        empty.append(d)

    

        
    
    
            
