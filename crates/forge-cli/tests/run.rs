use forge::run_agent;
use forge_core::agent::FakeAgent;
use forge_storage::{MemoryStorage, Storage};

#[tokio::test]
async fn fake_agent_writes_a_linked_chain() {
    let storage = MemoryStorage::new();
    let mut agent = FakeAgent::scripted();
    let chain = run_agent(&mut agent, &storage).await.expect("run ok");
    assert!(chain.len() >= 3, "expected a multi-step conversation");

    // First step is root.
    let root = storage.get(&chain[0]).await.unwrap().unwrap();
    assert!(root.parent.is_none());

    // Each subsequent step's parent is the previous step's id.
    for i in 1..chain.len() {
        let step = storage.get(&chain[i]).await.unwrap().unwrap();
        assert_eq!(step.parent.as_ref(), Some(&chain[i - 1]));
    }

    // Children index walks forward correctly.
    let kids = storage.children(&chain[0]).await.unwrap();
    assert_eq!(kids, vec![chain[1].clone()]);
}

#[tokio::test]
async fn run_is_deterministic_in_hashes() {
    let s1 = MemoryStorage::new();
    let s2 = MemoryStorage::new();
    let chain1 = run_agent(&mut FakeAgent::scripted(), &s1).await.unwrap();
    let chain2 = run_agent(&mut FakeAgent::scripted(), &s2).await.unwrap();
    assert_eq!(
        chain1, chain2,
        "same scripted agent should produce same hashes"
    );
}
