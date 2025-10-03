from github import Github
import github
from datetime import datetime
from github.PullRequest import PullRequest

import pickle
import os
import re
import datasets

import json

import github.TimelineEvent

from utils import logger 
import subprocess
import git
from git import PathLike

from functools import wraps
from inspect import signature
from typing import Literal, TypedDict, NewType, Sequence, Mapping, Any

import typing
if typing.TYPE_CHECKING:
    from issue_postpro import IssueWithBodyAnalysisDict

SPLIT_STR = "#@!@#"

# https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/using-keywords-in-issues-and-pull-requests
PR_KEYWORDS = {
    # official keywords used by github to auto-link issues. They are also used by SWE-bench
    "close",
    "closes",
    "closed",
    "fix",
    "fixes",
    "fixed",
    "resolve",
    "resolves",
    "resolved",
}
LOOSE_PR_KEYWORDS = PR_KEYWORDS | {
    # the following are not in the official doc, but some people use them
    "close issue",
    "closes issue",
    "closed issue",
    "fix issue",
    "fixes issue",
    "fixed issue",
    "resolve issue",
    "resolves issue",
    "resolved issue",
    "close the issue",
    "closes the issue",
    "closed the issue",
    "fix the issue",
    "fixes the issue",
    "fixed the issue",
    "resolve the issue",
    "resolves the issue",
    "resolved the issue",
    "solve",
    "solves",
    "solved",
    "solve issue",
    "solves issue",
    "solved issue",
    "solve the issue",
    "solves the issue",
    "solved the issue",
}

class User(TypedDict):
    login: str # The username of the user
    is_contributor: bool # Whether the user is a contributor of the repo
    is_member: bool # Whether the user is a member of the repo
    
class PRParticipantsDict(TypedDict):
    open: list[User] # The users who opened the pull request
    comment: list[User] # The users who commented on the pull request
    mentioned: list[User] # The users who were mentioned in the pull request
    merge: list[User] # The users who merged the pull request
    review_comment: list[User] # The users who made review comments on the pull request
    close: list[User] # The users who closed the pull request
    reopen: list[User] # The users who reopened the pull request
    rename: list[User] # The users who renamed the pull request
    assignee: list[User] # The users who are assigned to the pull request
    assigner: list[User] # The users who assigned the pull request
    review_requester: list[User] # The users who requested a review for the pull request
    requested_reviewer: list[User] # The users who are requested to review the pull request
    reviewer: list[User] # The users who reviewed the pull request
    labeled: list[User] # The users who labeled the pull request

class CommitDict(TypedDict):
    sha: str # The sha of the commit
    base: str # The base commit sha of the pull request
    parents: list[str] # The parents of the commit
    author: str | None # The author of the commit. It is possible that the author is None.
    committer: str | None # The committer of the commit. It is possible that the committer is None.
    files: list[str] # The files changed in the commit
    patches: list[str] # The file patches of the commit
    files_from_base: list[str] # The diff file names from the base of the pr.
    patches_from_base: list[str] # The diff file patches from the base of the pr.

class IssueDict(TypedDict):
    number: int # get the issue by github.get_issue(number=issue_index)
    title: str # issue.title
    body: str # issue.body, contain the issue description provided by the user
    labels: str # issue.labels, separated by ","
    comments: str # issue.get_comments(), separated by ";" Notice that the comments should not be used in prediction because there could be data leakage.
    
class ReviewDict(TypedDict):
    number: int # The review number
    body: str # The review body
    state: str # The review state, like "APPROVED", "CHANGES_REQUESTED", "COMMENTED", "DISMISSED"
    user: str # The user who made the review
    commit_id: str # The based commit sha of the review. The info of the commit can be found in the commit dict.
    submitted_at: datetime # The time when the review was submitted

class ReviewCommentDict(TypedDict): # refer to both review comment and commit comment
    id: int # The id of a review comment
    body: str # The body of the review comment
    base_commit_id: str # The base commit sha of the review comment, it could be a commit in another branch, but it can be found in the commit dict.
    in_reply: int # The id of the review comment that this comment is in reply to
    user: User # The user who made the review comment
    create_time: datetime # The time when the review comment was created
    updated_time: datetime # The time when the review comment was updated
    file: str # The file name of the review comment
    position: int # The position of the review comment in the diff. Based on the final commited file. If the code is not in the final commited file, it will be None and this review comment will be labeled as `outdated` in the github page. Closing down notice. The line number in the final diff file.
    original_position: int # The position of the review comment in the diff. Based on the file in the commit this comment is made. The line number in the this time diff file. (I guess it is the patches from base.))
    line: int # The line number of the review comment in the file. Based on the final commited file. If the code is not in the final commited file, it will be None and this review comment will be labeled as `outdated` in the github page. Closing down notice.
    original_line: int # The line number of the review comment in the file. Based on the file in the commit this comment is made. Closing down notice.

IssueInfoStr = NewType("IssueInfoStr", str)
# The string representation of information of some issues, concatenated by SPLIT_STR. Each part is a json string of IssueDict. Str is suitable for saving in the dataset.

PRState = Literal["open", "closed", "merged", "unknown"]

class PRDict(TypedDict):
    number: int
    title: str
    state: PRState
    body: str # The body of the pull request
    base_commit_sha: str # The base commit sha of the pull request. Switch to the base commit to reproduce the code version of the pull request.
    timestamp: datetime | None # The timestamp of the pull request. It is the merged time.
    files: str # The files changed in the pull request
    file_patches: str # The file patches of the pull request, concatenated by SPLIT_STR
    commit: str # The commits of the pull request, json dumps str of list[CommitDict], note that these commits could be on another branch (like a fork), which cannot be directly find in the repo.
    review_comment: str # The review comments of the pull request, separated by ";"
    comment: str # The comments of the pull request, separated by ";"
    review: str # The reviews of the pull request, separated by ";"
    issues: list[int] # The issues referenced by the pull request
    issues_info: IssueInfoStr # The issues information, concatenated by SPLIT_STR
    participants: str # json dumps str of PRParticipantsDict. The participants of the pull request.

def issue_dict_to_str(issue_dicts: Sequence[IssueDict] | Sequence["IssueWithBodyAnalysisDict"]) -> IssueInfoStr:
    return IssueInfoStr(SPLIT_STR.join([json.dumps(issue_dict) for issue_dict in issue_dicts]))

def issue_str_to_dict(issue_str: IssueInfoStr) -> list[IssueDict]:
    return [json.loads(_) for _ in issue_str.split(SPLIT_STR)]

def catch_exception(func):
    """
    Catch any exception in the function and log the error message, and return None.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # find possible "pr" input in args or kwargs, given the function signature
            signiture = signature(func)
            if "pr" in kwargs:
                pr = kwargs["pr"]
            elif "pr" in signiture.parameters:
                pr_index = list(signiture.parameters).index("pr")
                pr = args[pr_index]
            else:
                pr = None
                
            if pr is not None:
                if isinstance(pr, int):
                    number = pr
                    title = "Not available"
                else:
                    try:
                        number = pr.number
                        title = pr.title
                    except:
                        number = "Not available"
                        title = "Not available"
                logger.error(f"When process PR #{number}: {title}: Error in {func.__name__} with args: {args}, kwargs: {kwargs}. Exception: {e}")
            else:
                logger.error(f"Error in {func.__name__} with args: {args}, kwargs: {kwargs}. Exception: {e}")
            return None
    return wrapper

def user_table(save_path: str) -> None:
    """
    Post-process: collect all users appeared in a repo, save it as a json list. 
    """
    pass
    

class GithubAnalyzer:
    def __init__(self, repo_onwer: str, repo_name: str, access_token: str | None = None):
        access_token = access_token if access_token is not None else os.getenv("GITHUB_ACCESS_TOKEN")
        if access_token is None:
            raise ValueError("Please provide the github access token")
        self.access_token = access_token
        self.g = Github(access_token)
        print("begin init!!!", flush=True)
        self.repo = self.g.get_repo(f"{repo_onwer}/{repo_name}")
        print("end init!!!", flush=True)
        self.repo_name = repo_name
        self.repo_owner = repo_onwer
        self._members = None
        self._contributors = None
        
    def is_owned_by_organization(self) -> bool:
        """
        Check whether the repo is owned by an organization or a user.
        """
        return self.repo.owner.type == "Organization"
    
    def get_organization_members_tmp(self) -> list[str]:
        """
        Because the python api has bug to read the organization members, we need to use the curl command to get the members. Only use it temporarily.
        
        
        curl -L \
        -H "Accept: application/vnd.github+json" \
        -H "Authorization: Bearer <YOUR-TOKEN>" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        https://api.github.com/orgs/ORG/members
        """
        import subprocess
        command = f"curl -L -H \"Accept: application/vnd.github+json\" -H \"Authorization: Bearer {self.access_token}\" -H \"X-GitHub-Api-Version: 2022-11-28\" https://api.github.com/orgs/{self.repo_owner}/members?per_page=100&page=5"
        result = subprocess.check_output(command, shell=True)
        return [user["login"] for user in json.loads(result)]
        
    @property
    def members(self) -> list[str]:
        if self._members is None:
            if self.is_owned_by_organization():
                # If the repo is owned by an organization, get the members of the organization
                try:
                    self._members = [user.login for user in self.repo.organization.get_members()]
                except github.GithubException:
                    self._members = self.get_organization_members_tmp()
            else:
                raise ValueError("The repo is not owned by an organization, we cannot get the member information.")
        return self._members
    
    @property
    def contributors(self) -> list[str]:
        if self._contributors is None:
            self._contributors = [user.login for user in self.repo.get_contributors()]
        return self._contributors
    
    def from_login_to_user(self, login: str) -> User:
        """
        Convert the login name to a user object.
        """
        return {
            "login": login,
            "is_contributor": login in self.contributors,
            "is_member": login in self.members
        }

class GithubPRAnalyzer(GithubAnalyzer):
    
    def __init__(self, repo_onwer: str, repo_name: str, access_token: str | None = None, loose_match: bool = True): 
        
        super().__init__(repo_onwer=repo_onwer, repo_name=repo_name, access_token=access_token)
        self.pr_keywords = LOOSE_PR_KEYWORDS if loose_match else PR_KEYWORDS # use to determien the connection between pull request and linked issues. In github, if the text like "fix #1" is in the pull request, the issue #1 will be linked to the pull request. The issue #1 will be closed when the pull request is merged.

    # def extract_resolved_issues(self, pull: PullRequest) -> list[int]:

    #     # Define 1. issue number regex pattern 2. comment regex pattern 3. keywords
    #     # the issures should be PR_KEYWORDS + any white space + # + any number 
    #     issues_pat = re.compile(r"(?i)(?:\b(?:{})\b\s*#(\d+))".format("|".join(self.pr_keywords)))
    #     comments_pat = re.compile(r"(?s)<!--.*?-->")

    #     # Construct text to search over for issue numbers from PR body and commit messages
    #     text = pull.title if pull.title else ""
    #     text += "\n" + (pull.body if pull.body else "")
    #     commits = pull.get_commits()
    #     commit_messages = [commit.commit.message for commit in commits]
    #     commit_text = "\n".join(commit_messages) if commit_messages else ""
    #     text += "\n" + commit_text
    #     # Remove comments from text
    #     text = comments_pat.sub("", text)
    #     # Look for issue numbers in text via scraping <keyword, number> patterns
    #     references = issues_pat.findall(text.lower())
    #     logger.debug(f"References: {references}")
    #     resolved_issues = list()
    #     if references:
    #         for issue_num in references:
    #             if issue_num != pull.number and int(issue_num) not in resolved_issues:
    #                 resolved_issues.append(int(issue_num))
    #     return resolved_issues
    
    def extract_resolved_issues(self, pull: PullRequest) -> list[int]:
        
        original_pat = re.compile(r"(?i)(?:\b(?:{})\b\s*#(\d+))".format("|".join(self.pr_keywords)))
        
        
        simple_pat = re.compile(r"#(\d+)")
        
        
        url_pat = re.compile(r"(?i)github\.com/.*?/(?:issues|pull)/(\d+)")
        
        
        comments_pat = re.compile(r"(?s)<!--.*?-->")

        
        text = pull.title if pull.title else ""
        text += "\n" + (pull.body if pull.body else "")
        commits = pull.get_commits()
        commit_messages = [commit.commit.message for commit in commits]
        commit_text = "\n".join(commit_messages) if commit_messages else ""
        text += "\n" + commit_text
        
        
        text = comments_pat.sub("", text)
        
        
        original_refs = original_pat.findall(text.lower())  
        simple_refs = simple_pat.findall(text)              
        url_refs = url_pat.findall(text)                    
        
        logger.debug(f"Original references: {original_refs}")
        logger.debug(f"Simple references: {simple_refs}")
        logger.debug(f"URL references: {url_refs}")
        
        
        all_references = set(original_refs + simple_refs + url_refs)
        logger.debug(f"All references: {all_references}")
        
        resolved_issues = []
        if all_references:
            for issue_num in all_references:
                
                if issue_num != str(pull.number) and int(issue_num) not in resolved_issues:
                    resolved_issues.append(int(issue_num))
        
        return resolved_issues

    def check_approved(self, pr: PullRequest) -> bool:
        for r in pr.get_reviews():
            if r.state == "APPROVED":
                logger.info(f"Find approved in pr # {pr.number}: {pr.title}")
                # Find an approved review, break to process the pull request
                return True
        else:
            logger.info(f"Do NOT find approved in pr # {pr.number}: {pr.title}")
            return False
    
    def check_merged(self, pr: PullRequest) -> bool:
        if pr.merged:
            logger.info(f"Merged pr # {pr.number}: {pr.title}")
        return pr.merged

    def check_closed(self, pr: PullRequest) -> bool:
        if pr.closed_at is not None:
            logger.info(f"Closed pr # {pr.number}: {pr.title}")
        return pr.closed_at is not None

    def get_manual_link_issues(self, pr_merged_commit_sha, events):
        """If an issue is not automatically linked to the pr but manually linked to a pr, we need to find it by an 'connected' event. Rare case."""
        logger.warning("The function `get_manual_link_issues` has not been tested.")
        manual_link_issue = []
        for event in events:
            if event.event == "connected":
                logger.debug("Debug event", event.event, event.commit_id, event.issue.number, event.issue.title) 
                # check the issue is not a pull request
                issue = self.repo.get_issue(number=event.issue.number)
                if issue.pull_request:
                    logger.debug("Debug connected issue is a pull request, skip")
                    continue
                # check whether the issue is closed by this pull request
                for inner_event in issue.get_events():
                    if inner_event.event == "closed" and inner_event.commit_id == pr_merged_commit_sha and event.issue.number not in manual_link_issue:
                        logger.debug("Debug connected issue is closed by this pull request")
                        manual_link_issue.append(event.issue.number)
        return manual_link_issue
    
    def get_pr_state(self, pr: PullRequest) -> PRState:
        """
        Get the state of the pull request.
        """
        if pr.state == "open":
            return "open"
        else:
            if pr.merged:
                return "merged"
            elif pr.closed_at is not None:
                return "closed"
            else:
                return "unknown"
            
    def get_participants(self, pr: PullRequest) -> PRParticipantsDict:
        """
        Get the participants of the pull request.
        """
        participants: PRParticipantsDict = {
            "open": [],
            "comment": [],
            "mentioned": [],
            "merge": [],
            "review_comment": [],
            "close": [],
            "reopen": [],
            "rename": [],
            "assignee": [],
            "assigner": [],
            "review_requester": [],
            "requested_reviewer": [],
            "reviewer": [],
            "labeled": [],
        }
        if pr.user:
            participants["open"].append(self.from_login_to_user(pr.user.login))
        for comment in pr.get_issue_comments():
            participants["comment"].append(self.from_login_to_user(comment.user.login))
        for review_comment in pr.get_review_comments():
            participants["review_comment"].append(self.from_login_to_user(review_comment.user.login))
        if pr.merged_by:
            participants["merge"].append(self.from_login_to_user(pr.merged_by.login))
        for review in pr.get_reviews():
            participants["reviewer"].append(self.from_login_to_user(review.user.login))
        
        events = pr.get_issue_events()
        for event in events:
            if event.event == "closed":
                print(event.event, event.actor.login, event.commit_id, event)
            if event.event == "mentioned":
                participants["mentioned"].append(self.from_login_to_user(event.actor.login))
            elif event.event == "assigned":
                participants["assignee"].append(self.from_login_to_user(event.assignee.login))
                participants["assigner"].append(self.from_login_to_user(event.assigner.login))
            elif event.event == "review_requested":
                participants["requested_reviewer"].append(self.from_login_to_user(event.requested_reviewer.login))
                participants["review_requester"].append(self.from_login_to_user(event.actor.login))
            elif event.event == "closed":
                participants["close"].append(self.from_login_to_user(event.actor.login))
            elif event.event == "reopened":
                participants["reopen"].append(self.from_login_to_user(event.actor.login))
            elif event.event == "renamed":
                participants["rename"].append(self.from_login_to_user(event.actor.login))
            elif event.event == "labeled":
                participants["labeled"].append(self.from_login_to_user(event.actor.login))

        # use timeline events
        # from github.PaginatedList import PaginatedList
        # from github.Consts import mediaTypeLockReasonPreview
        # timeline_events = PaginatedList(
        #     github.TimelineEvent.TimelineEvent,
        #     pr._requester,
        #     f"{pr.issue_url}/timeline",
        #     None,
        #     headers={"Accept": mediaTypeLockReasonPreview},
        # )
        # for event in timeline_events:
        #     if event.event == "reviewed":
        #         participants["reviewer"].append(self.from_login_to_user(event.actor.login))
            
        # remove duplicate users
        for key in participants.keys():
            participants[key] = list({user["login"]: user for user in participants[key]}.values())
        return participants
    
    def process_pr_commits(self, pr: PullRequest) -> list[CommitDict]:
        base = pr.base.sha
        commits = pr.get_commits()
        return_list: list[CommitDict] = []
        for commit in commits:
            diff_from_base = self.repo.compare(base, commit.sha).files
            diff_from_parents = commit.files
            commit_info: CommitDict = {
                "sha": commit.sha,
                "parents": [_.sha for _ in commit.parents],
                "base": base,
                "author": commit.author.login if commit.author else None,
                "committer": commit.committer.login if commit.committer else None,
                "files": [f.filename for f in diff_from_parents],
                "patches": [f.patch for f in diff_from_parents],
                "files_from_base": [f.filename for f in diff_from_base],
                "patches_from_base": [f.patch for f in diff_from_base],
            }
            return_list.append(commit_info)
        return return_list
    
    def process_pr_reviews(self, pr: PullRequest) -> list[ReviewDict]:
        """
        """
        return_list: list[ReviewDict] = []
        reviews = pr.get_reviews()
        for review in reviews:
            review_dict: ReviewDict = {
                "number": review.id,
                "body": review.body,
                "state": review.state,
                "user": review.user.login,
                "commit_id": review.commit_id,
                "submitted_at": review.submitted_at
            }
            return_list.append(review_dict)
        return return_list
    
    def process_pr_review_comments(self, pr: PullRequest) -> list[ReviewCommentDict]:
        review_comments = pr.get_review_comments()
        return_list: list[ReviewCommentDict] = []
        for rc in review_comments:
            review_comment_dict: ReviewCommentDict = {
                "id": rc.id,
                "body": rc.body,
                "base_commit_id": rc.original_commit_id,
                "in_reply": rc.in_reply_to_id,
                "user": self.from_login_to_user(rc.user.login),
                "create_time": rc.created_at,
                "updated_time": rc.updated_at,
                "file": rc.path,
                "original_line": rc.original_line,
                "line": rc.line,
                "original_position": rc.original_position,
                "position": rc.position
            }
            return_list.append(review_comment_dict)
        return return_list

    # TODO: maybe add more filters, like determine whether the pr has changed at least one .py file, or it should be done by a post-procession
    @catch_exception
    def process_pull_request(self, pr: PullRequest | int, keep: Literal["merged", "closed", "approved", "all"], commits2file: None | Mapping[str, Sequence[PathLike]] = None, with_issue: bool = True, only_main_branch: bool = True) -> PRDict | None:
        """
        Given a pr or pr number, retrieve some information by github api including some basic information of the pr as well as the related issues. The issue information is stored in the "issues_info" field and saved as a concatenated json string with the `SPLIT_STR` as the separator. It can be further processed by the `issue_postprocess` function.
        ------
        pr: PullRequest | int
            The pull request object or the pull request number.
        keep: Literal["merged", "closed", "approved", "all"]
            The pull requests to keep.
        commits2file: dict[str, list[PathLike]]
            The commit to file dictionary, used to filter out the pull requests that are not in the main branch.
        """
        
        if isinstance(pr, int):
            # receive an int as the pull request number, used for multiprocessing, where we cannot pass the PullRequest object directly because some api problem.
            pr = self.repo.get_pull(number=pr)
        assert isinstance(pr, PullRequest)
        
        print(f"Start processing # {pr.number}: {pr.title}", flush=True)
        
        if not self.check_merged(pr):
            logger.info(f"[Skip] pr # {pr.number}: {pr.title}. Skip reason: Not merged.")
            return
        '''
        if keep == "all":
            pass
        elif keep == "approved":
            if not self.check_approved(pr):
                logger.info(f"[Skip] pr # {pr.number}: {pr.title}. Skip reason: Not approved.")
                return
        elif keep == "merged":
            if not self.check_merged(pr):
                logger.info(f"[Skip] pr # {pr.number}: {pr.title}. Skip reason: Not merged.")
                return
        elif keep == "closed":
            if not self.check_closed(pr):
                logger.info(f"[Skip] pr # {pr.number}: {pr.title}. Skip reason: Not closed.")   
                return
        else:
            raise ValueError("Invalid value for keep")
        '''
        # events = pr.get_issue_events()
        files = pr.get_files()
        file_names  = [file.filename for file in files]
        file_patches = [file.patch for file in files]
        if None in file_patches:
            print(f"[Skip] `None` is found in patches of files. pr #{pr.number}: {pr.title}. For now on, we have no idea why it happens. See https://github.com/PyGithub/PyGithub/issues/3270.", flush=True)
            return
        
        issue_numbers = self.extract_resolved_issues(pr)
        # if with_issue and not issue_numbers:
        #     print(f"[Skip] pr # {pr.number}: {pr.title}. Skip reason: No solved issue found in it.", flush=True)
        #     return 
        

        base = pr.base.sha # base is the commit that the pull request is based on. It is the commit that the pull request will be merged into. Even if there could be multiple parents for a merge commit, the base is only one.
        print(f"base: {base}", flush=True)
        # commits
        # commits = [_.sha for _ in pr.get_commits()] 
        commits = self.process_pr_commits(pr)
        
        # reviews
        reviews = self.process_pr_reviews(pr)
        
        # timestamp = pr.last_modified_datetime
        timestamp = pr.merged_at # The timestamp is the merged time
        review_comments = [_.body for _ in pr.get_comments()] # The "get_comments" method only returns review comments, not issue comments
        comments = [_.body for _ in pr.get_issue_comments()] # The "get_issue_comments" method returns chat comments in the pr issue, not for the related issues
        reviews = [_.body for _ in pr.get_review_comments()]
        
        # get issue info for each issue number
        issues_list: list[IssueDict] = []
        for issue_index in issue_numbers:
            issue = self.repo.get_issue(number=issue_index)
            # issue labels is a list of Label objects. These labels are general used to indicate the feature or states of the issue. Common labels are "bug", "feature request", "enhancement","document", etc. We collect them to further analyze the issue category.
            issue_labels = ", ".join([label.name for label in issue.labels])
            issue_dict: IssueDict = {"number": issue.number, "title": issue.title, "body": issue.body, "labels": issue_labels, "comments": ";".join([_.body for _ in issue.get_comments()])}
            issues_list.append(issue_dict)
        issues_str = issue_dict_to_str(issues_list)
        
        participants = self.get_participants(pr)
        

        output_dict: PRDict = {"number": pr.number, "title": pr.title, "state": self.get_pr_state(pr),"body": pr.body, "base_commit_sha": base, "timestamp": timestamp, "files":";".join(file_names), "file_patches": SPLIT_STR.join(file_patches), "commit": json.dumps(commits), "review_comment": ";".join(review_comments), "comment": ";".join(comments), "review": ";".join(reviews), "issues": [-1] + issue_numbers, "issues_info": issues_str, "participants": json.dumps(participants)}
        print(f"[Collect] pr #{pr.number}: {pr.title}", flush=True)
        return output_dict

    def process(self, keep: Literal["merged", "closed", "approved", "all"] = "merged", with_issue: bool = True, only_main_branch: bool = True, cpu_count: int = 4, test_parquet_path: str = None) -> list[PRDict | None]:
        """
        Process all the closed PR in a repo.
        ------
        keep: Literal["merged", "closed", "approved", "all"]
            The pull requests to keep.
        with_issue: bool
            If True, only keep the pr linked by at least one issue. It is useful when the task is code fixing.
        only_main_branch: bool
            If True, only keep the pull requests in the main branch.
        cpu_count: int
            The number of cpus to use. If it is less than 1, it will use the cpu_count of the machine.
        ------
        Return:
            A list of PRDict or None. The None is used to indicate the pull request is skipped.
        """
        print("begin process!!!", flush=True)
        # Fetch only closed pull requests
        pull_requests = self.repo.get_pulls(state="closed", sort="created", direction="asc")
        #pull_requests = self.repo.get_pulls(state=("approved" if keep == "approved" else "all"), direction="asc")
        #pull_requests = self.repo.get_pulls(state=("merged" if keep == "merged" else "all"), direction="asc")
        print("iter process!!!", flush=True)
        #pull_requests = list(pull_requests)
        #print(f"Total {len(pull_requests)} pull requests found in the repo {self.repo_name}.", flush=True)
        repo_path = f'repos/{self.repo_name}'
        try:
            print(subprocess.check_output(f"cd {repo_path}; git checkout -f main", shell=True))
        except Exception:
            try:
                print(subprocess.check_output(f"cd {repo_path}; git checkout -f master", shell=True))
            except Exception:
                try:
                    print(subprocess.check_output(f"cd {repo_path}; git checkout -f pre-commit-ci-update-config", shell=True))
                except Exception:
                    print(subprocess.check_output(f"cd {repo_path}; git checkout -f develop2", shell=True))
        from commit_utils import CommitDAGAnalyzer
        analyzer = CommitDAGAnalyzer(repo_name=self.repo_name)
        
        
        if test_parquet_path and os.path.exists(test_parquet_path):
            print(f"Loading SHA from test parquet: {test_parquet_path}")
            import pandas as pd
            df = pd.read_parquet(test_parquet_path)
            
            current_repo = f"{self.repo_owner}/{self.repo_name}"
            repo_data = df[df['repo'] == current_repo]
            if len(repo_data) > 0:
                sha_path = repo_data['base_commit'].unique().tolist()
                print(f"Found {len(sha_path)} unique SHAs for {current_repo} in test data")
            else:
                print(f"No data found for {current_repo} in test parquet, using default longest path")
                sha_path = analyzer.get_longest_path()
        else:
            sha_path = analyzer.get_longest_path()
            
        storage_dir = "pulldata"  # specify your desired folder
        
        os.makedirs(storage_dir, exist_ok=True)
        
        suffix = "_test" if test_parquet_path and os.path.exists(test_parquet_path) else ""
        output_filepath = os.path.join(storage_dir, f"{self.repo_name}_sha_path{suffix}.pkl")
        with open(output_filepath, "wb") as f:
            pickle.dump(sha_path, f)
        
        print("iter filter!!!", flush=True)

        batch_size = 50
        all_results = []
        temp_dir = os.path.join(storage_dir, f"{self.repo_name}_temp{suffix}")
        os.makedirs(temp_dir, exist_ok=True)
        
        
        processed_batches = set()
        for filename in os.listdir(temp_dir):
            if filename.startswith("batch_") and filename.endswith(".arrow"):
                batch_num = int(filename.split("_")[1].split(".")[0])
                processed_batches.add(batch_num)
        processed_batches = max(processed_batches) if processed_batches else -1
        try:
            for i, pr in enumerate(pull_requests):
                      
                batch_num = i // batch_size
                
                
                if batch_num <= processed_batches:
                    print(f"Skipping already processed batch {batch_num}")
                    continue
                if pr.base.sha not in sha_path:
                    continue    
                
                result = self.process_pull_request(pr, keep, commits2file=sha_path, with_issue=with_issue, only_main_branch=only_main_branch)
                if result is not None:
                    all_results.append(result)
                
                
                if (i + 1) % batch_size == 0 :
                    if all_results:
                        batch_dataset = datasets.Dataset.from_list(all_results)
                        batch_path = os.path.join(temp_dir, f"batch_{batch_num}.arrow")
                        batch_dataset.save_to_disk(batch_path)
                        print(f"Saved batch {batch_num} with {len(all_results)} PRs")
                        all_results = []
            if all_results:
                        batch_dataset = datasets.Dataset.from_list(all_results)
                        batch_path = os.path.join(temp_dir, f"batch_{batch_num}.arrow")
                        batch_dataset.save_to_disk(batch_path)
                        print(f"Saved batch {batch_num} with {len(all_results)} PRs")
                        all_results = []
        except KeyboardInterrupt:
            print("\n\033[1;33mKeyboard interrupt detected! Saving current batch...\033[0m")
            if all_results:
                
                current_batch_num = (i // batch_size) if i >= 0 else 0
                batch_path = os.path.join(temp_dir, f"batch_{current_batch_num}.arrow")
                datasets.Dataset.from_list(all_results).save_to_disk(batch_path)
                print(f"Saved current batch {current_batch_num} with {len(all_results)} PRs")
        
        
        all_batches = []
        for filename in os.listdir(temp_dir):
            if filename.startswith("batch_") and filename.endswith(".arrow"):
                batch_path = os.path.join(temp_dir, filename)
                try:
                    batch_dataset = datasets.load_from_disk(batch_path)
                    all_batches.append(batch_dataset)
                    print(f"Loaded batch {filename} with {len(batch_dataset)} PRs")
                except Exception as e:
                    print(f"Error loading batch {filename}: {e}")
        
        if all_batches:
            full_dataset = datasets.concatenate_datasets(all_batches)
            full_path = os.path.join(args.save_path, self.repo_name)
            full_dataset.save_to_disk(full_path)
            print(f"Saved full dataset to {full_path} with {len(full_dataset)} PRs")
            
            
            #for filename in os.listdir(temp_dir):
            #    os.remove(os.path.join(temp_dir, filename))
            #os.rmdir(temp_dir)
            return full_dataset
        else:
            print("No PRs processed")
            return []

    
def parse_args():
    import argparse
    import pickle
    import os
    parser = argparse.ArgumentParser(description="Github Repo Analyzer")
    parser.add_argument("-o", "--repo_owner", type=str, help="The owner of the repository")
    parser.add_argument("-n", "--repo_name", type=str, help="The name of the repository")
    parser.add_argument("-t", "--token", type=str, help="The access token for the github", default="github_pat_xxx")
    parser.add_argument("--keep", type=str, default="merged", choices=["merged", "closed", "approved", "all"], help="The pull requests to keep")
    parser.add_argument("--keep_wo_issue", action="store_true", help="Whether to include the issues")
    parser.add_argument("--keep_non_main_branch", action="store_true", help="Whether to only include the pull requests in the main branch")
    parser.add_argument("-c", "--cpu", type=int, default=1, help="The number of cpus to use")
    parser.add_argument("--save_path", type=str, default="pulldata", help="The path to save the pull data")
    parser.add_argument("--test_parquet", type=str, default=None, help="Path to test parquet file with specific SHAs to use")
    return parser.parse_args()
    
if __name__ == "__main__":
    #import multiprocessing
    #multiprocessing.set_start_method('spawn')
    args = parse_args()
    processer = GithubPRAnalyzer(
        repo_onwer=args.repo_owner,
        repo_name=args.repo_name,
        access_token=args.token,
    )
    pulldata = processer.process(keep=args.keep, with_issue=not args.keep_wo_issue, only_main_branch=not args.keep_non_main_branch, cpu_count=args.cpu, test_parquet_path=args.test_parquet)
    pulldata = [pr for pr in pulldata if pr is not None]
    dataset = datasets.Dataset.from_list(pulldata) # type: ignore
    dataset.save_to_disk(os.path.join(args.save_path, args.repo_name))

# def issue_gen():
#     issues = repo.get_issues(state="closed", sort="created-asc")
#     try:
#         for i, issue in enumerate(issues):
#             try:
#                 comments = [_.body for _ in issue.get_comments()]
#                 yield {"number": issue.number, "title": issue.title, "comment": ";".join(comments)}
#             except:
#                 break
#     except:
#         pass

# issuedata = datasets.Dataset.from_generator(issue_gen)
# issuedata.save_to_disk(f"issuedata/{REPO_NAME}")


# for i in range(len(os.listdir(f"issues/{REPO_NAME}/"))):
#     with open(f"issues/{REPO_NAME}/{i}.pkl", "rb") as f:
#         issue: Issue = pickle.load(f)
#         if issue.pull_request:
#             print(issue)
#             print(issue.number, issue.closed_at, issue.created_at, [(_.body, _.created_at, _) for _ in issue.get_comments()])#, issue.create_comment())