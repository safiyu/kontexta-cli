import os
import pytest
from kontexta_cli.context.repomap import RepoMap

def test_repomap_treesitter_and_diskcache(tmp_path):
    RepoMap.invalidate_cache()
    # TypeScript file
    (tmp_path / "service.ts").write_text("""
export class AuthService extends BaseService {
    async login(user: string, pass: string): Promise<boolean> { return true; }
}
export function validateToken(token: string) {}
""")
    # Rust file
    (tmp_path / "main.rs").write_text("""
struct User { id: u64 }
fn process_user(u: User) {}
""")
    # Python file
    (tmp_path / "app.py").write_text("""
class App:
    def run(self): pass
def main(): pass
""")

    md = RepoMap.get_cached_map(str(tmp_path))
    assert "service.ts" in md
    assert "class AuthService" in md
    assert "validateToken" in md
    assert "main.rs" in md
    assert "process_user" in md
    assert "app.py" in md
    assert "class App" in md

    # Verify diskcache
    stats = RepoMap.cache_stats()
    assert stats["cached"] is True
    assert stats["disk_cached"] is True

    # Check fast cache hit
    second = RepoMap.get_cached_map(str(tmp_path))
    assert second == md
