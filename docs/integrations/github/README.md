# GitHub account tools

`web_search` `source=github` searches public repositories. It does not create or change them. `shell_exec` can run `git` or `gh` only after approval, and it is off unless `SOPHON_SHELL_TOOLS=1`.

These tools talk to the GitHub REST API with a token. They do not `git init` a folder on another drive.

## Env

```powershell
$env:SOPHON_GITHUB_TOKEN = "github_pat_..."
$env:SOPHON_GITHUB_TOOLS = "1"
```

`GH_TOKEN` and `GITHUB_TOKEN` are also read. `SOPHON_GITHUB_TOOLS=0` hides the tools.

## Tools

| Tool | Role |
| --- | --- |
| `github_status` | Token present, authenticated login |
| `github_repos` | Repos for that user |
| `github_repo` | One `owner/name` |
| `github_create` | Create a remote repo. Harness asks first. No local checkout |

`/github` is the same surface: `status`, `repos`, `repo NAME`, `create NAME`.

## References

- [GitHub REST: create a repository for the authenticated user](https://docs.github.com/en/rest/repos/repos#create-a-repository-for-the-authenticated-user)
- [GitHub REST: list repositories for the authenticated user](https://docs.github.com/en/rest/repos/repos#list-repositories-for-the-authenticated-user)
