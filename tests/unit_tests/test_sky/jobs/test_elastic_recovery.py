"""Unit tests for the ELASTIC managed job recovery strategy."""
# pylint: disable=protected-access
from typing import Dict, List, Optional, Sequence
from unittest import mock

import pytest

from sky import backends
from sky import exceptions
from sky import resources as resources_lib
from sky import task as task_lib
from sky.jobs import recovery_strategy


def _make_task(num_nodes: int, node_counts: Optional[List[int]]):
    job_recovery = {'strategy': 'ELASTIC'}
    if node_counts is not None:
        job_recovery['node_counts'] = node_counts
    task = task_lib.Task(run='true', num_nodes=num_nodes)
    task.set_resources(resources_lib.Resources(job_recovery=job_recovery))
    return task


_DEFAULT_NODE_COUNTS = (4, 2, 1)


def _make_executor(num_nodes: int = 4,
                   node_counts: Optional[Sequence[int]] = _DEFAULT_NODE_COUNTS,
                   pool: Optional[str] = None):
    task = _make_task(num_nodes,
                      list(node_counts) if node_counts is not None else None)
    backend = mock.Mock(spec=backends.CloudVmRayBackend)
    return recovery_strategy.StrategyExecutor.make('cluster', backend, task, 1,
                                                   0, pool, set(), mock.Mock(),
                                                   mock.Mock())


class _FakeLaunch:
    """Stands in for _launch: outcome per attempt, keyed by num_nodes.

    Each outcome is a list consumed one attempt at a time; the last entry
    repeats. 'ok' succeeds, 'capacity' raises
    ManagedJobReachedMaxRetriesError, 'precheck' raises
    ProvisionPrechecksError.
    """

    def __init__(self, executor, outcomes: Dict[int, List[str]]):
        self._executor = executor
        self._outcomes = outcomes
        self.attempts: List[int] = []

    async def __call__(self,
                       max_retry=3,
                       raise_on_failure=True,
                       recovery=False):
        del raise_on_failure, recovery  # Unused.
        assert max_retry == 1
        n = self._executor.dag.tasks[0].num_nodes
        self.attempts.append(n)
        seq = self._outcomes[n]
        outcome = seq.pop(0) if len(seq) > 1 else seq[0]
        if outcome == 'ok':
            return 100.0
        if outcome == 'capacity':
            raise exceptions.ManagedJobReachedMaxRetriesError('no capacity')
        assert outcome == 'precheck', outcome
        raise exceptions.ProvisionPrechecksError(
            reasons=[ValueError(f'{n} nodes infeasible')])


def _install(executor, outcomes, monkeypatch):
    fake = _FakeLaunch(executor, outcomes)
    executor._launch = fake
    executor._cleanup_cluster = mock.MagicMock()
    sleep = mock.AsyncMock()
    monkeypatch.setattr(recovery_strategy.asyncio, 'sleep', sleep)
    return fake, sleep


def test_make_selects_elastic_and_reads_node_counts():
    executor = _make_executor(num_nodes=4, node_counts=[4, 2, 1])
    assert isinstance(executor, recovery_strategy.ElasticStrategyExecutor)
    assert executor._node_counts == [4, 2, 1]
    # Nothing ELASTIC-specific is persisted in the task specs.
    assert executor.task_specs() == {}
    # job_recovery is stripped from the task's resources by make().
    assert all(r.job_recovery is None for r in executor.dag.tasks[0].resources)


@pytest.mark.parametrize(
    'num_nodes,node_counts,match',
    [
        (4, None, 'requires `node_counts`'),
        (4, [4, 4, 1], 'strictly descending'),
        (4, [4, 1, 2], 'strictly descending'),
        (4, [2, 1], 'must equal the task\'s num_nodes'),
        (4, [], 'non-empty list'),
    ],
)
def test_invalid_node_counts_rejected(num_nodes, node_counts, match):
    with pytest.raises(ValueError, match=match):
        _make_executor(num_nodes=num_nodes, node_counts=node_counts)


def test_pool_rejected():
    with pytest.raises(ValueError, match='pool'):
        _make_executor(pool='my-pool')


def test_schema_rejects_non_positive_node_counts():
    with pytest.raises(ValueError):
        task_lib.Task.from_yaml_config({
            'run': 'true',
            'num_nodes': 2,
            'resources': {
                'job_recovery': {
                    'strategy': 'ELASTIC',
                    'node_counts': [2, 0],
                }
            },
        })


def test_node_counts_survive_yaml_round_trip():
    task = task_lib.Task.from_yaml_config({
        'run': 'true',
        'num_nodes': 4,
        'resources': {
            'job_recovery': {
                'strategy': 'elastic',
                'node_counts': [4, 2],
            }
        },
    })
    reparsed = task_lib.Task.from_yaml_config(task.to_yaml_config())
    (resources,) = reparsed.resources
    assert resources.job_recovery == {
        'strategy': 'ELASTIC',
        'node_counts': [4, 2]
    }


@pytest.mark.asyncio
async def test_launch_uses_largest_available(monkeypatch):
    executor = _make_executor()
    fake, sleep = _install(executor, {
        4: ['capacity'],
        2: ['ok'],
        1: ['ok']
    }, monkeypatch)
    assert await executor.launch() == 100.0
    assert fake.attempts == [4, 2]
    assert executor.dag.tasks[0].num_nodes == 2
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_launch_skips_infeasible_size(monkeypatch):
    executor = _make_executor()
    fake, _ = _install(executor, {
        4: ['precheck'],
        2: ['ok'],
        1: ['ok']
    }, monkeypatch)
    assert await executor.launch() == 100.0
    assert fake.attempts == [4, 2]


@pytest.mark.asyncio
async def test_recover_retries_largest_first(monkeypatch):
    executor = _make_executor()
    # The first launch only gets 2 nodes; at recovery, 4 are available.
    fake, _ = _install(executor, {
        4: ['capacity', 'ok'],
        2: ['ok'],
        1: ['ok']
    }, monkeypatch)
    await executor.launch()
    assert executor.dag.tasks[0].num_nodes == 2
    await executor.recover()
    executor._cleanup_cluster.assert_called_once()
    assert fake.attempts == [4, 2, 4]
    assert executor.dag.tasks[0].num_nodes == 4


@pytest.mark.asyncio
async def test_backs_off_when_no_size_fits(monkeypatch):
    executor = _make_executor()
    fake, sleep = _install(executor, {
        4: ['capacity'],
        2: ['capacity'],
        1: ['capacity', 'ok']
    }, monkeypatch)
    assert await executor.recover() == 100.0
    assert fake.attempts == [4, 2, 1, 4, 2, 1]
    sleep.assert_awaited_once()
    assert executor.dag.tasks[0].num_nodes == 1


@pytest.mark.asyncio
async def test_raises_when_every_size_fails_prechecks(monkeypatch):
    executor = _make_executor()
    fake, sleep = _install(executor, {
        4: ['precheck'],
        2: ['precheck'],
        1: ['precheck']
    }, monkeypatch)
    with pytest.raises(exceptions.ProvisionPrechecksError) as exc_info:
        await executor.launch()
    assert len(exc_info.value.reasons) == 3
    assert fake.attempts == [4, 2, 1]
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_keeps_retrying_when_some_sizes_are_infeasible(monkeypatch):
    executor = _make_executor()
    fake, sleep = _install(executor, {
        4: ['precheck'],
        2: ['capacity', 'ok'],
        1: ['capacity']
    }, monkeypatch)
    assert await executor.launch() == 100.0
    assert fake.attempts == [4, 2, 1, 4, 2]
    sleep.assert_awaited_once()


@pytest.mark.asyncio
async def test_on_resume_restores_launched_node_count(monkeypatch):
    executor = _make_executor()
    handle = mock.Mock(spec=backends.CloudVmRayResourceHandle)
    handle.launched_nodes = 2
    monkeypatch.setattr(recovery_strategy.global_user_state,
                        'get_handle_from_cluster_name', lambda name: handle)
    monkeypatch.setattr(recovery_strategy.state,
                        'get_runtime_user_restarts_async',
                        mock.AsyncMock(return_value=0))
    await executor.on_resume('cluster')
    assert executor.dag.tasks[0].num_nodes == 2


def test_elastic_is_registered_but_not_default():
    registry = recovery_strategy.registry.JOBS_RECOVERY_STRATEGY_REGISTRY
    assert registry.from_str('ELASTIC') is (
        recovery_strategy.ElasticStrategyExecutor)
    assert registry.default == 'eager_next_region'
