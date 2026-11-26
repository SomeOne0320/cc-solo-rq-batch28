import os
import threading
from datetime import datetime, timedelta, timezone
from multiprocessing import Process
from unittest import mock

import pytest
import redis
from redis.client import Pipeline
from redis.exceptions import WatchError

from rq import Queue
from rq.defaults import DEFAULT_MAINTENANCE_TASK_INTERVAL
from rq.exceptions import NoSuchJobError, SchedulerNotFound
from rq.job import Job, JobStatus, Retry
from rq.rate_limit import RateLimit, RateLimitRegistry
from rq.registry import FinishedJobRegistry, ScheduledJobRegistry
from rq.scheduler import RQScheduler
from rq.serializers import JSONSerializer
from rq.utils import current_timestamp, now, utcformat
from rq.worker import Worker
from tests import RQTestCase, find_empty_redis_database, ssl_test, system_timezone

from .fixtures import CustomJob, CustomQueue, kill_worker, say_hello


class CustomRedisConnection(redis.Connection):
    """Custom redis connection with a custom arg, used in test_custom_connection_pool"""

    def __init__(self, *args, custom_arg=None, **kwargs):
        self.custom_arg = custom_arg
        super().__init__(*args, **kwargs)

    def get_custom_arg(self):
        return self.custom_arg


class TestScheduledJobRegistry(RQTestCase):
    def test_get_jobs_to_enqueue(self):
        """Getting job ids to enqueue from ScheduledJobRegistry."""
        queue = Queue(connection=self.connection)
        registry = ScheduledJobRegistry(queue=queue)
        timestamp = current_timestamp()

        self.connection.zadd(registry.key, {'foo': 1})
        self.connection.zadd(registry.key, {'bar': timestamp + 10})
        self.connection.zadd(registry.key, {'baz': timestamp + 30})

        self.assertEqual(registry.get_jobs_to_enqueue(), ['foo'])
        self.assertEqual(registry.get_jobs_to_enqueue(timestamp + 20), ['foo', 'bar'])

    def test_get_jobs_to_schedule_with_chunk_size(self):
        """Max amount of jobs returns by get_jobs_to_schedule() equal to chunk_size"""
        queue = Queue(connection=self.connection)
        registry = ScheduledJobRegistry(queue=queue)
        timestamp = current_timestamp()
        chunk_size = 5

        for index in range(0, chunk_size * 2):
            self.connection.zadd(registry.key, {f'foo_{index}': 1})

        self.assertEqual(len(registry.get_jobs_to_schedule(timestamp, chunk_size)), chunk_size)
        self.assertEqual(len(registry.get_jobs_to_schedule(timestamp, chunk_size * 2)), chunk_size * 2)

    def test_get_scheduled_time(self):
        """get_scheduled_time() returns job's scheduled datetime"""
        queue = Queue(connection=self.connection)
        registry = ScheduledJobRegistry(queue=queue)

        job = Job.create('myfunc', connection=self.connection)
        job.save()
        dt = datetime(2019, 1, 1, tzinfo=timezone.utc)
        registry.schedule(job, datetime(2019, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(registry.get_scheduled_time(job), dt)
        # get_scheduled_time() should also work with job ID
        self.assertEqual(registry.get_scheduled_time(job.id), dt)

        # registry.get_scheduled_time() raises NoSuchJobError if
        # job.id is not found
        self.assertRaises(NoSuchJobError, registry.get_scheduled_time, '123')

    def test_schedule(self):
        """Adding job with the correct score to ScheduledJobRegistry"""
        queue = Queue(connection=self.connection)
        job = Job.create('myfunc', connection=self.connection)
        job.save()
        registry = ScheduledJobRegistry(queue=queue)

        # A naive datetime is local time, at the UTC offset in effect on that date:
        # New York is UTC-5 in January and UTC-4 in July. Unique scheduling converts it the same way.
        with system_timezone('America/New_York'):
            for scheduled_time, timestamp in [
                (datetime(2019, 1, 1), 1546300800 + 5 * 3600),
                (datetime(2019, 7, 1), 1561939200 + 4 * 3600),
            ]:
                registry.schedule(job, scheduled_time)
                self.assertEqual(self.connection.zscore(registry.key, job.id), timestamp)

                unique_job = Job.create('myfunc', id=f'unique-{scheduled_time.month}', connection=self.connection)
                queue.schedule_job(unique_job, scheduled_time, unique=True)
                self.assertEqual(self.connection.zscore(registry.key, unique_job.id), timestamp)

        # Score is always stored in UTC even if datetime is in a different tz
        tz = timezone(timedelta(hours=7))
        job = Job.create('myfunc', connection=self.connection)
        job.save()
        registry.schedule(job, datetime(2019, 1, 1, 7, tzinfo=tz))
        self.assertEqual(self.connection.zscore(registry.key, job.id), 1546300800)  # 2019-01-01 UTC in Unix timestamp

    def test_remove_jobs(self):
        """Removing job ids from ScheduledJobRegistry. Will be deprecated in the future."""
        queue = Queue(connection=self.connection)
        registry = ScheduledJobRegistry(queue=queue)
        timestamp = current_timestamp()

        self.connection.zadd(registry.key, {'foo': 1})
        self.connection.zadd(registry.key, {'bar': timestamp + 10})
        self.connection.zadd(registry.key, {'baz': timestamp + 30})

        with pytest.deprecated_call():
            # without timestamp it should remove everything till the current timestamp
            registry.remove_jobs()
            self.assertListEqual(registry.get_jobs_to_schedule(timestamp=timestamp + 100), ['bar', 'baz'])
            registry.remove_jobs(timestamp=timestamp + 15)
            # should remove bar job
            self.assertListEqual(registry.get_jobs_to_schedule(timestamp=timestamp + 100), ['baz'])


class TestScheduler(RQTestCase):
    def test_init(self):
        """Scheduler can be instantiated with queues or queue names"""
        foo_queue = Queue('foo', connection=self.connection)
        scheduler = RQScheduler([foo_queue, 'bar'], connection=self.connection)
        self.assertEqual(scheduler._queue_names, {'foo', 'bar'})
        self.assertEqual(scheduler.status, RQScheduler.Status.STOPPED)

    def test_name(self):
        """Scheduler generates a unique name and derives its hash key from it"""
        scheduler = RQScheduler(['foo'], connection=self.connection)
        self.assertTrue(scheduler.name)
        self.assertEqual(scheduler.key, f'rq:scheduler:{scheduler.name}')

        # An explicitly provided name is honored
        named = RQScheduler(['foo'], connection=self.connection, name='my-scheduler')
        self.assertEqual(named.name, 'my-scheduler')

    def test_register_birth_and_fetch(self):
        """register_birth() writes a metadata hash that fetch() round-trips"""
        scheduler = RQScheduler(['foo', 'bar'], connection=self.connection)
        scheduler.register_birth()

        data = self.connection.hgetall(scheduler.key)
        self.assertEqual(int(data[b'pid']), os.getpid())
        self.assertIsNotNone(scheduler.last_heartbeat)
        # The hash carries a TTL
        self.assertTrue(scheduler.interval + 55 <= self.connection.ttl(scheduler.key) <= scheduler.interval + 60)

        fetched = RQScheduler.fetch(scheduler.name, self.connection)
        self.assertEqual(fetched.name, scheduler.name)
        self.assertEqual(fetched.hostname, scheduler.hostname)
        self.assertEqual(fetched.pid, scheduler.pid)
        self.assertEqual(fetched._queue_names, {'foo', 'bar'})
        self.assertEqual(fetched.created_at, scheduler.created_at)
        self.assertEqual(fetched.last_heartbeat, scheduler.last_heartbeat)

        self.connection.delete(scheduler.key)
        scheduler.save()
        self.assertEqual(self.connection.hget(scheduler.key, 'name').decode(), scheduler.name)

    def test_register_birth_is_idempotent(self):
        """register_birth() can be called again (e.g. on restart) without error"""
        scheduler = RQScheduler(['foo'], connection=self.connection)
        scheduler.register_birth()
        scheduler.register_birth()  # must not raise
        self.assertTrue(self.connection.exists(scheduler.key))

    def test_fetch_missing_scheduler_raises(self):
        """fetch() raises SchedulerNotFound when no metadata hash exists"""
        with self.assertRaises(SchedulerNotFound):
            RQScheduler.fetch('does-not-exist', self.connection)

    def test_register_death(self):
        """register_death() returns whether the metadata hash was deleted"""
        scheduler = RQScheduler(['foo'], connection=self.connection)
        scheduler.register_birth()
        self.assertTrue(self.connection.exists(scheduler.key))

        self.assertTrue(scheduler.register_death())
        self.assertFalse(self.connection.exists(scheduler.key))
        self.assertFalse(scheduler.register_death())

    def test_should_reacquire_locks(self):
        """scheduler.should_reacquire_locks works properly"""
        queue = Queue(connection=self.connection)
        scheduler = RQScheduler([queue], connection=self.connection)
        self.assertTrue(scheduler.should_reacquire_locks)
        scheduler.acquire_locks()
        self.assertIsNotNone(scheduler.lock_acquisition_time)

        # scheduler.should_reacquire_locks always returns False if
        # scheduler.acquired_locks and scheduler._queue_names are the same
        self.assertFalse(scheduler.should_reacquire_locks)
        scheduler.lock_acquisition_time = datetime.now() - timedelta(seconds=DEFAULT_MAINTENANCE_TASK_INTERVAL + 6)
        self.assertFalse(scheduler.should_reacquire_locks)

        scheduler._queue_names = {'default', 'foo'}
        self.assertTrue(scheduler.should_reacquire_locks)
        scheduler.acquire_locks()
        self.assertFalse(scheduler.should_reacquire_locks)

    def test_lock_acquisition(self):
        """Test lock acquisition"""
        name_1 = 'lock-test-1'
        name_2 = 'lock-test-2'
        name_3 = 'lock-test-3'
        scheduler = RQScheduler([name_1], self.connection)

        self.assertEqual(scheduler.acquire_locks(), {name_1})
        self.assertEqual(scheduler._acquired_locks, {name_1})
        # A second call refreshes the lock held by the same scheduler
        self.assertEqual(scheduler.acquire_locks(), {name_1})

        # Only name_2 is returned since name_1 is locked by another scheduler
        scheduler = RQScheduler([name_1, name_2], self.connection)
        self.assertEqual(scheduler.acquire_locks(), {name_2})
        self.assertEqual(scheduler._acquired_locks, {name_2})

        # name_2 is refreshed while name_3 is newly acquired
        scheduler._queue_names.add(name_3)
        self.assertEqual(scheduler.acquire_locks(), {name_2, name_3})
        self.assertEqual(scheduler._acquired_locks, {name_2, name_3})

    def test_lock_acquisition_drops_taken_locks(self):
        """acquire_locks() drops locks overwritten by another scheduler and leaves them untouched"""
        name = 'lock-test-taken'
        scheduler = RQScheduler([name], self.connection)
        locking_key = scheduler.get_locking_key(name)

        self.assertEqual(scheduler.acquire_locks(), {name})
        self.connection.set(locking_key, 'other-scheduler', ex=5)

        self.assertEqual(scheduler.acquire_locks(), set())
        self.assertEqual(scheduler._acquired_locks, set())

        # The other scheduler's token is still in place and its TTL was not extended or removed
        self.assertEqual(self.connection.get(locking_key), b'other-scheduler')
        ttl = self.connection.ttl(locking_key)
        self.assertGreater(ttl, 0)
        self.assertLessEqual(ttl, 5)

        # With no verified locks, auto_start doesn't fork a scheduler process
        scheduler.acquire_locks(auto_start=True)
        self.assertIsNone(scheduler._process)

    def test_lock_acquisition_refreshes_own_token(self):
        """acquire_locks() keeps a lock still holding this scheduler's token and refreshes its TTL"""
        name = 'lock-test-refresh'
        scheduler = RQScheduler([name], self.connection)
        locking_key = scheduler.get_locking_key(name)

        self.assertEqual(scheduler.acquire_locks(), {name})
        self.connection.expire(locking_key, 5)

        self.assertEqual(scheduler.acquire_locks(), {name})
        self.assertEqual(scheduler._acquired_locks, {name})
        self.assertGreaterEqual(self.connection.ttl(locking_key), 55)

    def test_lock_acquisition_with_auto_start(self):
        """Test lock acquisition with auto_start=True"""
        scheduler = RQScheduler(['auto-start'], self.connection)
        with mock.patch.object(scheduler, 'start') as mocked:
            scheduler.acquire_locks(auto_start=True)
            self.assertEqual(mocked.call_count, 1)

        # If process has started, scheduler.start() won't be called
        running_process = mock.MagicMock()
        running_process.is_alive.return_value = True
        scheduler = RQScheduler(['auto-start2'], self.connection)
        scheduler._process = running_process
        with mock.patch.object(scheduler, 'start') as mocked:
            scheduler.acquire_locks(auto_start=True)
            self.assertEqual(mocked.call_count, 0)
            self.assertEqual(running_process.is_alive.call_count, 1)

        # If the process has stopped for some reason, the scheduler should restart
        scheduler = RQScheduler(['auto-start3'], self.connection)
        stopped_process = mock.MagicMock()
        stopped_process.is_alive.return_value = False
        scheduler._process = stopped_process
        with mock.patch.object(scheduler, 'start') as mocked:
            scheduler.acquire_locks(auto_start=True)
            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(stopped_process.is_alive.call_count, 1)

    def test_lock_release(self):
        """Test that scheduler.release_locks() only releases acquired locks"""
        name_1 = 'lock-test-1'
        name_2 = 'lock-test-2'
        scheduler_1 = RQScheduler([name_1], self.connection)

        self.assertEqual(scheduler_1.acquire_locks(), {name_1})
        self.assertEqual(scheduler_1._acquired_locks, {name_1})

        # Only name_2 is returned since name_1 is already locked
        scheduler_1_2 = RQScheduler([name_1, name_2], self.connection)
        self.assertEqual(scheduler_1_2.acquire_locks(), {name_2})
        self.assertEqual(scheduler_1_2._acquired_locks, {name_2})

        self.assertTrue(self.connection.exists(scheduler_1.get_locking_key(name_1)))
        self.assertTrue(self.connection.exists(scheduler_1_2.get_locking_key(name_1)))
        self.assertTrue(self.connection.exists(scheduler_1_2.get_locking_key(name_2)))

        scheduler_1_2.release_locks()

        self.assertEqual(scheduler_1_2._acquired_locks, set())
        self.assertEqual(scheduler_1._acquired_locks, {name_1})

        self.assertTrue(self.connection.exists(scheduler_1.get_locking_key(name_1)))
        self.assertTrue(self.connection.exists(scheduler_1_2.get_locking_key(name_1)))
        self.assertFalse(self.connection.exists(scheduler_1_2.get_locking_key(name_2)))

    def test_queue_scheduler_pid(self):
        queue = Queue(connection=self.connection)
        scheduler = RQScheduler([queue], connection=self.connection)
        scheduler.acquire_locks()

        # The lock value is the scheduler's name
        self.assertEqual(self.connection.get(scheduler.get_locking_key(queue.name)).decode(), scheduler.name)
        # Until register_birth() writes the metadata hash, the pid is unknown
        self.assertIsNone(queue.scheduler_pid)

        scheduler.register_birth()
        assert queue.scheduler_pid == os.getpid()

    def test_heartbeat(self):
        """Test that heartbeat updates the locking keys' TTL and the scheduler's metadata"""
        name_1 = 'lock-test-1'
        name_2 = 'lock-test-2'
        name_3 = 'lock-test-3'
        scheduler = RQScheduler([name_3], self.connection)
        scheduler.acquire_locks()
        scheduler = RQScheduler([name_1, name_2, name_3], self.connection)
        scheduler.acquire_locks()
        scheduler.register_birth()

        locking_key_1 = RQScheduler.get_locking_key(name_1)
        locking_key_2 = RQScheduler.get_locking_key(name_2)
        locking_key_3 = RQScheduler.get_locking_key(name_3)

        with self.connection.pipeline() as pipeline:
            pipeline.expire(locking_key_1, 1000)
            pipeline.expire(locking_key_2, 1000)
            pipeline.expire(locking_key_3, 1000)
            pipeline.execute()
        self.connection.expire(scheduler.key, 1000)

        scheduler.heartbeat()
        self.assertEqual(self.connection.ttl(locking_key_1), 61)
        self.assertEqual(self.connection.ttl(locking_key_2), 61)
        self.assertEqual(self.connection.ttl(locking_key_3), 1000)

        # heartbeat() also advances last_heartbeat and refreshes the metadata hash TTL
        self.assertIsNotNone(scheduler.last_heartbeat)
        stored = self.connection.hget(scheduler.key, 'last_heartbeat')
        self.assertEqual(stored.decode(), utcformat(scheduler.last_heartbeat))
        self.assertTrue(scheduler.interval + 55 <= self.connection.ttl(scheduler.key) <= scheduler.interval + 60)

        # scheduler.stop() releases locks, sets status to STOPPED, and deletes the metadata hash
        scheduler._status = scheduler.Status.WORKING
        scheduler.stop()
        self.assertFalse(self.connection.exists(locking_key_1))
        self.assertFalse(self.connection.exists(locking_key_2))
        self.assertTrue(self.connection.exists(locking_key_3))
        self.assertEqual(scheduler.status, scheduler.Status.STOPPED)
        self.assertFalse(self.connection.exists(scheduler.key))

        # Heartbeat also works properly for schedulers with a single queue
        scheduler = RQScheduler([name_1], self.connection)
        scheduler.acquire_locks()
        self.connection.expire(locking_key_1, 1000)
        scheduler.heartbeat()
        self.assertEqual(self.connection.ttl(locking_key_1), 61)

    def test_heartbeat_updates_acquired_locks(self):
        """heartbeat() keeps owned and expired locks but drops locks taken by another scheduler.
        Redis-level lock effects are covered by the script tests in test_scripts.py."""
        scheduler = RQScheduler(['hb-owned', 'hb-expired', 'hb-taken'], self.connection)
        scheduler.acquire_locks()
        scheduler.register_birth()
        scheduler.prepare_registries()

        self.connection.delete(scheduler.get_locking_key('hb-expired'))
        self.connection.set(scheduler.get_locking_key('hb-taken'), 'another-scheduler', ex=5)

        scheduler.heartbeat()
        self.assertEqual(scheduler._acquired_locks, {'hb-owned', 'hb-expired'})
        self.assertEqual(scheduler._scheduled_job_registries, [])

        scheduler.release_locks()
        scheduler.register_death()

    def test_release_locks_leaves_taken_lock(self):
        """release_locks() does not delete a lock this scheduler no longer owns"""
        name = 'release-taken'
        scheduler = RQScheduler([name], self.connection)
        scheduler.acquire_locks()
        locking_key = scheduler.get_locking_key(name)

        # The lock changes owners while this scheduler still believes it owns it
        self.connection.set(locking_key, 'another-scheduler', ex=5)

        scheduler.release_locks()
        self.assertEqual(scheduler._acquired_locks, set())
        self.assertEqual(self.connection.get(locking_key), b'another-scheduler')

    def test_work_heartbeats_before_enqueuing(self):
        """work() verifies locks via heartbeat() before enqueuing scheduled jobs"""
        scheduler = RQScheduler(['work-loop-order'], self.connection)
        scheduler.acquire_locks()
        calls = []

        def stop_after_first_iteration(interval):
            scheduler._stop_requested = True

        with (
            mock.patch.object(scheduler, 'heartbeat', side_effect=lambda: calls.append('heartbeat')),
            mock.patch.object(scheduler, 'enqueue_scheduled_jobs', side_effect=lambda: calls.append('enqueue')),
            mock.patch('rq.scheduler.time.sleep', side_effect=stop_after_first_iteration),
        ):
            scheduler.work()

        self.assertEqual(calls, ['heartbeat', 'enqueue'])
        # stop() ran: locks released and the metadata hash deleted
        self.assertEqual(scheduler._acquired_locks, set())
        self.assertFalse(self.connection.exists(scheduler.key))

    def test_enqueue_scheduled_jobs(self):
        """Scheduler can enqueue scheduled jobs"""
        queue = Queue(connection=self.connection)
        registry = ScheduledJobRegistry(queue=queue)
        job = Job.create('myfunc', connection=self.connection)
        job.save()
        job.set_status(JobStatus.SCHEDULED)
        registry.schedule(job, datetime(2019, 1, 1, tzinfo=timezone.utc))
        scheduler = RQScheduler([queue], connection=self.connection)
        scheduler.acquire_locks()
        scheduler.enqueue_scheduled_jobs()
        self.assertEqual(len(queue), 1)

        # After job is scheduled, registry should be empty
        self.assertEqual(len(registry), 0)

        # Jobs scheduled in the far future should not be affected
        job.set_status(JobStatus.SCHEDULED)
        registry.schedule(job, datetime(2100, 1, 1, tzinfo=timezone.utc))
        scheduler.enqueue_scheduled_jobs()
        self.assertEqual(len(queue), 1)

    def test_prepare_registries(self):
        """prepare_registries() creates self._scheduled_job_registries"""
        foo_queue = Queue('foo', connection=self.connection)
        bar_queue = Queue('bar', connection=self.connection)
        scheduler = RQScheduler([foo_queue, bar_queue], connection=self.connection)
        self.assertEqual(scheduler._scheduled_job_registries, [])
        scheduler.prepare_registries([foo_queue.name])
        self.assertEqual(scheduler._scheduled_job_registries, [ScheduledJobRegistry(queue=foo_queue)])
        scheduler.prepare_registries([foo_queue.name, bar_queue.name])
        self.assertEqual(
            scheduler._scheduled_job_registries,
            [ScheduledJobRegistry(queue=foo_queue), ScheduledJobRegistry(queue=bar_queue)],
        )


class TestScheduledJobRegistryEnqueue(RQTestCase):
    """ScheduledJobRegistry.enqueue_jobs() must transfer jobs based on the status and score
    valid at transfer time, never on a stale candidate list."""

    def setUp(self):
        super().setUp()
        self.queue = Queue(connection=self.connection)
        self.registry = ScheduledJobRegistry(queue=self.queue)
        self.past = datetime(2019, 1, 1, tzinfo=timezone.utc)

    def _schedule_due_job(self, job_id='due-job', **kwargs):
        job = self.queue.enqueue_at(self.past, say_hello, job_id=job_id, **kwargs)
        self.assertEqual(job.get_status(), JobStatus.SCHEDULED)
        return job

    def test_due_job_is_enqueued_and_removed(self):
        """A still-valid due job is enqueued (QUEUED, at-front honored) and leaves the registry."""
        sentinel = self.queue.enqueue(say_hello, job_id='sentinel')
        job = self._schedule_due_job(job_id='front-job', at_front=True)

        enqueued = self.registry.enqueue_jobs([job.id], queue=self.queue)

        self.assertEqual(enqueued, [job])
        self.assertNotIn(job.id, self.registry.get_job_ids(cleanup=False))
        self.assertEqual(Job.fetch(job.id, connection=self.connection).get_status(), JobStatus.QUEUED)
        self.assertEqual(self.queue.get_job_ids(), [job.id, sentinel.id])

    def test_canceled_job_between_read_and_transfer_stays_canceled(self):
        """A job canceled after the candidate read is not enqueued; its stale entry is dropped
        and the CANCELED status (and CanceledJobRegistry membership) is preserved."""
        from rq.registry import CanceledJobRegistry

        job = self._schedule_due_job(job_id='canceled-midflight')
        job_ids = self.registry.get_jobs_to_schedule()

        # Operator cancels between the scheduler's read and its transfer.
        job.cancel()
        self.assertEqual(job.get_status(), JobStatus.CANCELED)

        enqueued = self.registry.enqueue_jobs(job_ids, queue=self.queue)

        self.assertEqual(enqueued, [])
        self.assertEqual(self.queue.get_job_ids(), [])
        self.assertNotIn(job.id, self.registry.get_job_ids(cleanup=False))
        self.assertEqual(Job.fetch(job.id, connection=self.connection).get_status(), JobStatus.CANCELED)
        self.assertIn(job.id, CanceledJobRegistry(queue=self.queue).get_job_ids())

    def test_rescheduled_job_waits_for_new_time(self):
        """A job pushed to a later score after the candidate read is not transferred early."""
        job = self._schedule_due_job(job_id='rescheduled-midflight')
        job_ids = self.registry.get_jobs_to_schedule()
        new_time = now() + timedelta(hours=1)

        self.queue.schedule_job(job, new_time)

        enqueued = self.registry.enqueue_jobs(job_ids, queue=self.queue)

        self.assertEqual(enqueued, [])
        self.assertEqual(self.queue.get_job_ids(), [])
        self.assertEqual(Job.fetch(job.id, connection=self.connection).get_status(), JobStatus.SCHEDULED)
        # Both the entry and the new (future) score are untouched.
        self.assertIn(job.id, self.registry.get_job_ids(cleanup=False))
        self.assertGreater(self.connection.zscore(self.registry.key, job.id), current_timestamp())

        # Once the new time arrives, the same job transfers normally.
        self.registry.schedule(job, self.past)
        enqueued = self.registry.enqueue_jobs([job.id], queue=self.queue)
        self.assertEqual([j.id for j in enqueued], [job.id])
        self.assertEqual(self.queue.get_job_ids(), [job.id])

    def test_deleted_job_entry_is_cleaned_but_siblings_transfer(self):
        """A dangling entry (job record deleted) is cleaned like before, without affecting the
        other due jobs in the same batch."""
        deleted = self._schedule_due_job(job_id='deleted-midflight')
        survivor = self._schedule_due_job(job_id='survivor')
        job_ids = self.registry.get_jobs_to_schedule()

        deleted.delete()
        self.assertFalse(self.connection.exists(Job.key_for(deleted.id)))

        enqueued = self.registry.enqueue_jobs(job_ids, queue=self.queue)

        self.assertEqual([j.id for j in enqueued], [survivor.id])
        self.assertNotIn(deleted.id, self.registry.get_job_ids(cleanup=False))
        self.assertNotIn(survivor.id, self.registry.get_job_ids(cleanup=False))
        self.assertEqual(self.queue.get_job_ids(), [survivor.id])

    def test_non_scheduled_status_entry_is_dropped_without_enqueue(self):
        """An entry whose job's status is no longer SCHEDULED is dropped, never enqueued."""
        job = self._schedule_due_job(job_id='status-changed')
        # Simulate another scheduler having enqueued it already: QUEUED but entry still present.
        job.set_status(JobStatus.QUEUED)

        enqueued = self.registry.enqueue_jobs([job.id], queue=self.queue)

        self.assertEqual(enqueued, [])
        self.assertNotIn(job.id, self.registry.get_job_ids(cleanup=False))
        # No duplicate push onto the queue.
        self.assertEqual(self.queue.get_job_ids(), [])

    def test_watcherror_means_job_enters_queue_at_most_once(self):
        """If the watched keys change during transfer, EXEC aborts: nothing is enqueued and the
        entry stays for the next tick."""
        job = self._schedule_due_job(job_id='contended')

        original_execute = Pipeline.execute

        def fake_execute(self_pipe, *args, **kwargs):
            if getattr(self_pipe, 'watching', False):
                raise WatchError('simulated scheduler contention')
            return original_execute(self_pipe, *args, **kwargs)

        with mock.patch('redis.client.Pipeline.execute', fake_execute):
            enqueued = self.registry.enqueue_jobs([job.id], queue=self.queue)

        self.assertEqual(enqueued, [])
        self.assertEqual(self.queue.get_job_ids().count(job.id), 0)
        self.assertIn(job.id, self.registry.get_job_ids(cleanup=False))
        self.assertEqual(Job.fetch(job.id, connection=self.connection).get_status(), JobStatus.SCHEDULED)

        # The next uncontended tick transfers it exactly once.
        enqueued = self.registry.enqueue_jobs([job.id], queue=self.queue)
        self.assertEqual([j.id for j in enqueued], [job.id])
        self.assertEqual(self.queue.get_job_ids(), [job.id])

    def test_two_schedulers_same_batch_enqueue_each_job_once(self):
        """Two schedulers acting on the same candidate IDs: the WATCH winner transfers the job,
        the loser's transaction aborts and never pushes a second copy."""
        job = self._schedule_due_job(job_id='double-scheduler')
        job_ids = [job.id]
        contender_ran = {'done': False}

        original_execute = Pipeline.execute

        def contender_execute(self_pipe, *args, **kwargs):
            # Right when the loser's watched transaction reaches EXEC, another scheduler commits
            # the same transfer first. The watched zset then changes, so Redis aborts this EXEC
            # with WatchError — a genuine read/transfer interleaving, no faked exception.
            if getattr(self_pipe, 'watching', False) and not contender_ran['done']:
                contender_ran['done'] = True
                winner = ScheduledJobRegistry(queue=self.queue)
                winner.enqueue_jobs(list(job_ids), queue=self.queue)
                self.assertEqual(self.queue.get_job_ids(), [job.id])
            return original_execute(self_pipe, *args, **kwargs)

        with mock.patch('redis.client.Pipeline.execute', contender_execute):
            enqueued = self.registry.enqueue_jobs(job_ids, queue=self.queue)

        self.assertEqual([j.id for j in enqueued], [])
        self.assertEqual(self.queue.get_job_ids(), [job.id])
        self.assertNotIn(job.id, self.registry.get_job_ids(cleanup=False))
        self.assertEqual(Job.fetch(job.id, connection=self.connection).get_status(), JobStatus.QUEUED)

    def test_cancel_observed_under_watch_drops_entry_inside_transaction(self):
        """If the status flips to CANCELED while the transfer is being watched, the re-read inside
        WATCH catches it and the entry is removed without enqueuing."""
        job = self._schedule_due_job(job_id='canceled-under-watch')

        original_hget = Pipeline.hget

        def fake_hget(self_pipe, name, key):
            if name == job.key and key == 'status':
                return JobStatus.CANCELED.value.encode()
            return original_hget(self_pipe, name, key)

        with mock.patch('redis.client.Pipeline.hget', fake_hget):
            enqueued = self.registry.enqueue_jobs([job.id], queue=self.queue)

        self.assertEqual(enqueued, [])
        self.assertEqual(self.queue.get_job_ids(), [])
        self.assertNotIn(job.id, self.registry.get_job_ids(cleanup=False))

    def test_later_score_observed_under_watch_skips_transfer(self):
        """If the score moves to the future while the transfer is being watched, the watched
        re-read skips the job and leaves the entry for the new due time."""
        job = self._schedule_due_job(job_id='rescheduled-under-watch')

        original_zscore = Pipeline.zscore

        def fake_zscore(self_pipe, key, member):
            if key == self.registry.key and member == job.id:
                return current_timestamp() + 3600
            return original_zscore(self_pipe, key, member)

        with mock.patch('redis.client.Pipeline.zscore', fake_zscore):
            enqueued = self.registry.enqueue_jobs([job.id], queue=self.queue)

        self.assertEqual(enqueued, [])
        self.assertEqual(self.queue.get_job_ids(), [])
        self.assertIn(job.id, self.registry.get_job_ids(cleanup=False))
        self.assertEqual(Job.fetch(job.id, connection=self.connection).get_status(), JobStatus.SCHEDULED)

    def test_rate_limited_job_follows_same_contract(self):
        """Due rate-limited jobs go through the rate limit path with the same transfer-time
        guarantees; slots and waiting-set membership are preserved."""
        rate_limit = RateLimit(key='scheduled-rl', concurrency=1)

        # Capacity free: transferred once, holding a slot and sitting on the queue.
        job = self._schedule_due_job(job_id='rl-due', rate_limit=rate_limit)
        enqueued = self.registry.enqueue_jobs([job.id], queue=self.queue)
        self.assertEqual([j.id for j in enqueued], [job.id])
        rl_registry = RateLimitRegistry(key='scheduled-rl', connection=self.connection)
        self.assertIn(job.id, rl_registry.get_allowed_job_ids())
        self.assertEqual(self.queue.get_job_ids(), [job.id])
        self.assertNotIn(job.id, self.registry.get_job_ids(cleanup=False))

        # Capacity exhausted: a second due job lands in the rate_limited waiting set, not the queue.
        waiting = self._schedule_due_job(job_id='rl-waiting', rate_limit=rate_limit)
        enqueued = self.registry.enqueue_jobs([waiting.id], queue=self.queue)
        self.assertEqual([j.id for j in enqueued], [waiting.id])
        self.assertNotIn(waiting.id, self.queue.get_job_ids())
        self.assertIn(waiting.id, rl_registry.get_rate_limited_job_ids())
        self.assertEqual(Job.fetch(waiting.id, connection=self.connection).get_status(), JobStatus.RATE_LIMITED)
        self.assertNotIn(waiting.id, self.registry.get_job_ids(cleanup=False))

        # A canceled rate-limited due job is never added to the waiting set.
        canceled = self._schedule_due_job(job_id='rl-canceled', rate_limit=rate_limit)
        canceled.cancel()
        enqueued = self.registry.enqueue_jobs([canceled.id], queue=self.queue)
        self.assertEqual(enqueued, [])
        self.assertNotIn(canceled.id, rl_registry.get_rate_limited_job_ids())

    def test_concurrent_transfers_enqueue_each_job_once(self):
        """Real parallel schedulers acting on the same candidate list: every job is enqueued
        exactly once and no transfer raises on an entry claimed by another scheduler."""
        job_count = 20
        job_ids = [self._schedule_due_job(job_id=f'race-{i}').id for i in range(job_count)]
        candidate = list(job_ids)  # every scheduler holds the same stale due list
        enqueued_counts: list[int] = []
        errors: list[Exception] = []

        def run_scheduler() -> None:
            registry = ScheduledJobRegistry(
                self.queue.name,
                connection=redis.Redis(db=self.connection.connection_pool.connection_kwargs['db']),
            )
            queue = Queue(
                self.queue.name,
                connection=redis.Redis(db=self.connection.connection_pool.connection_kwargs['db']),
            )
            try:
                enqueued_counts.append(len(registry.enqueue_jobs(list(candidate), queue=queue)))
            except Exception as exc:  # pragma: no cover - fails the test if a race escapes
                errors.append(exc)

        threads = [threading.Thread(target=run_scheduler) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(sum(enqueued_counts), job_count)
        self.assertEqual(sorted(self.queue.get_job_ids()), sorted(job_ids))
        self.assertEqual(len(set(self.queue.get_job_ids())), job_count)
        self.assertEqual(self.registry.get_job_count(cleanup=False), 0)


class TestSchedulerStaleListInterleaving(RQTestCase):
    """End-to-end scheduler ticks must honor transfer-time state for normal and rate-limited
    jobs, across queues."""

    def _scheduler(self, queue):
        # Locks carry a TTL and are otherwise still held by a previously built scheduler;
        # release them so the new instance actually owns the queue and schedules it.
        self.connection.delete(RQScheduler.get_locking_key(queue.name))
        scheduler = RQScheduler([queue], connection=self.connection)
        scheduler.acquire_locks()
        return scheduler

    def test_scheduler_skips_canceled_and_rescheduled_and_enqueues_rest(self):
        """One tick over a stale due list: canceled stays canceled, rescheduled waits for its new
        time, deleted entries are cleaned, and the remaining due jobs are enqueued."""
        queue = Queue('stale-tick', connection=self.connection)
        registry = ScheduledJobRegistry(queue=queue)
        past = datetime(2019, 1, 1, tzinfo=timezone.utc)

        good = queue.enqueue_at(past, say_hello, job_id='good')
        canceled = queue.enqueue_at(past, say_hello, job_id='canceled')
        rescheduled = queue.enqueue_at(past, say_hello, job_id='rescheduled')
        deleted = queue.enqueue_at(past, say_hello, job_id='deleted')

        # Take the candidate snapshot, then interleave operator actions before the tick transfers.
        job_ids = registry.get_jobs_to_schedule()
        self.assertEqual(set(job_ids), {'good', 'canceled', 'rescheduled', 'deleted'})

        canceled.cancel()
        queue.schedule_job(rescheduled, now() + timedelta(hours=2))
        deleted.delete()

        self._scheduler(queue).enqueue_scheduled_jobs()

        self.assertEqual(queue.get_job_ids(), [good.id])
        self.assertEqual(Job.fetch(good.id, connection=self.connection).get_status(), JobStatus.QUEUED)
        self.assertEqual(Job.fetch(canceled.id, connection=self.connection).get_status(), JobStatus.CANCELED)
        self.assertEqual(Job.fetch(rescheduled.id, connection=self.connection).get_status(), JobStatus.SCHEDULED)
        self.assertIn(rescheduled.id, registry.get_job_ids(cleanup=False))
        self.assertFalse(self.connection.exists(Job.key_for(deleted.id)))
        self.assertNotIn(deleted.id, registry.get_job_ids(cleanup=False))

        # The rescheduled job transfers when its new time arrives.
        registry.schedule(rescheduled, past)
        self._scheduler(queue).enqueue_scheduled_jobs()
        self.assertEqual(set(queue.get_job_ids()), {good.id, rescheduled.id})

    def test_scheduler_two_ticks_same_list_enqueue_each_job_once(self):
        """Two schedulers holding the same due list can each run a tick, but every job reaches the
        queue exactly once."""
        queue = Queue('double-tick', connection=self.connection)
        registry = ScheduledJobRegistry(queue=queue)
        past = datetime(2019, 1, 1, tzinfo=timezone.utc)
        jobs = [queue.enqueue_at(past, say_hello, job_id=f'job-{i}') for i in range(5)]

        scheduler_a = self._scheduler(queue)
        scheduler_b = self._scheduler(queue)
        scheduler_a.enqueue_scheduled_jobs()
        scheduler_b.enqueue_scheduled_jobs()

        self.assertEqual(sorted(queue.get_job_ids()), sorted(job.id for job in jobs))
        self.assertEqual(registry.get_job_count(cleanup=False), 0)

    def test_scheduler_preserves_queue_isolation(self):
        """Jobs due on different registries are pushed onto their own queues only."""
        alpha = Queue('alpha', connection=self.connection)
        beta = Queue('beta', connection=self.connection)
        past = datetime(2019, 1, 1, tzinfo=timezone.utc)
        job_a = alpha.enqueue_at(past, say_hello, job_id='a')
        job_b = beta.enqueue_at(past, say_hello, job_id='b')

        scheduler = RQScheduler([alpha, beta], connection=self.connection)
        scheduler.acquire_locks()
        scheduler.enqueue_scheduled_jobs()

        self.assertEqual(alpha.get_job_ids(), [job_a.id])
        self.assertEqual(beta.get_job_ids(), [job_b.id])
        self.assertEqual(ScheduledJobRegistry(queue=alpha).get_job_count(cleanup=False), 0)
        self.assertEqual(ScheduledJobRegistry(queue=beta).get_job_count(cleanup=False), 0)

    def test_scheduler_rate_limited_canceled_job_never_waits_nor_queues(self):
        """A rate-limited due job canceled before transfer keeps CANCELED and never reaches the
        waiting set or the queue, while a sibling due job still transfers."""
        queue = Queue('rl-stale-tick', connection=self.connection)
        registry = ScheduledJobRegistry(queue=queue)
        rate_limit = RateLimit(key='tick-rl', concurrency=1)
        past = datetime(2019, 1, 1, tzinfo=timezone.utc)

        sibling = queue.enqueue_at(past, say_hello, job_id='sibling', rate_limit=rate_limit)
        canceled = queue.enqueue_at(past, say_hello, job_id='canceled-rl', rate_limit=rate_limit)

        registry.get_jobs_to_schedule()  # stale snapshot
        canceled.cancel()

        self._scheduler(queue).enqueue_scheduled_jobs()

        rl_registry = RateLimitRegistry(key='tick-rl', connection=self.connection)
        self.assertEqual(queue.get_job_ids(), [sibling.id])
        self.assertIn(sibling.id, rl_registry.get_allowed_job_ids())
        self.assertNotIn(canceled.id, rl_registry.get_rate_limited_job_ids())
        self.assertNotIn(canceled.id, registry.get_job_ids(cleanup=False))
        self.assertEqual(Job.fetch(canceled.id, connection=self.connection).get_status(), JobStatus.CANCELED)


class TestWorker(RQTestCase):
    def test_work_burst(self):
        """worker.work() with scheduler enabled works properly"""
        queue = Queue(connection=self.connection)
        worker = Worker(queues=[queue], connection=self.connection)
        worker.work(burst=True, with_scheduler=False)
        self.assertIsNone(worker.scheduler)

        worker = Worker(queues=[queue], connection=self.connection)
        worker.work(burst=True, with_scheduler=True)
        assert worker.scheduler
        self.assertIsNone(self.connection.get(worker.scheduler.get_locking_key('default')))
        # The birth/death pair leaves no metadata hash behind
        self.assertFalse(self.connection.exists(worker.scheduler.key))

    def test_work_burst_scheduler_cleans_up_on_error(self):
        """If enqueue_scheduled_jobs() raises in burst, the lock and metadata hash are still released"""
        queue = Queue(connection=self.connection)
        worker = Worker(queues=[queue], connection=self.connection)
        with mock.patch.object(RQScheduler, 'enqueue_scheduled_jobs', side_effect=RuntimeError('boom')):
            with self.assertRaises(RuntimeError):
                worker.work(burst=True, with_scheduler=True)

        assert worker.scheduler
        self.assertIsNone(self.connection.get(worker.scheduler.get_locking_key('default')))
        self.assertFalse(self.connection.exists(worker.scheduler.key))

    @mock.patch.object(RQScheduler, 'acquire_locks')
    def test_run_maintenance_tasks(self, mocked):
        """scheduler.acquire_locks() is called only when scheduled is enabled"""
        queue = Queue(connection=self.connection)
        worker = Worker(queues=[queue], connection=self.connection)

        worker.run_maintenance_tasks()
        self.assertEqual(mocked.call_count, 0)

        # if scheduler object exists and it's a first start, acquire locks should not run
        worker.last_cleaned_at = None
        worker.scheduler = RQScheduler([queue], connection=self.connection)
        worker.run_maintenance_tasks()
        self.assertEqual(mocked.call_count, 0)

        # the scheduler exists and it's NOT a first start, since the process doesn't exists,
        # should call acquire_locks to start the process
        worker.last_cleaned_at = datetime.now()
        worker.run_maintenance_tasks()
        self.assertEqual(mocked.call_count, 1)

        # the scheduler exists, the process exists, but the process is not alive
        running_process = mock.MagicMock()
        running_process.is_alive.return_value = False
        worker.scheduler._process = running_process
        worker.run_maintenance_tasks()
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(running_process.is_alive.call_count, 1)

        # the scheduler exists, the process exits, and it is alive. acquire_locks shouldn't run
        running_process.is_alive.return_value = True
        worker.run_maintenance_tasks()
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(running_process.is_alive.call_count, 2)

    def test_work(self):
        queue = Queue(connection=self.connection)
        worker = Worker(queues=[queue], connection=self.connection)
        p = Process(target=kill_worker, args=(os.getpid(), False, 5))

        p.start()
        queue.enqueue_at(datetime(2019, 1, 1, tzinfo=timezone.utc), say_hello)
        worker.work(burst=False, with_scheduler=True)
        p.join(1)
        self.assertIsNotNone(worker.scheduler)
        registry = FinishedJobRegistry(queue=queue)
        self.assertEqual(len(registry), 1)

    @ssl_test
    def test_work_with_ssl(self):
        connection = find_empty_redis_database(ssl=True)
        queue = Queue(connection=connection)
        worker = Worker(queues=[queue], connection=connection)
        p = Process(target=kill_worker, args=(os.getpid(), False, 5))

        p.start()
        queue.enqueue_at(datetime(2019, 1, 1, tzinfo=timezone.utc), say_hello)
        worker.work(burst=False, with_scheduler=True)
        p.join(1)
        self.assertIsNotNone(worker.scheduler)
        registry = FinishedJobRegistry(queue=queue)
        self.assertEqual(len(registry), 1)

    def test_work_burst_with_job_class(self):
        """The scheduler fetches scheduled jobs with the worker's job_class"""
        with mock.patch.object(CustomJob, 'redis_job_namespace_prefix', 'custom:job:'):
            queue = Queue(connection=self.connection, job_class=CustomJob)
            job = queue.enqueue_at(datetime(2019, 1, 1, tzinfo=timezone.utc), say_hello)
            self.assertEqual(job.get_status(), JobStatus.SCHEDULED)

            worker = Worker([queue], connection=self.connection, job_class=CustomJob)
            worker.work(burst=True, with_scheduler=True)

            job.refresh()
            self.assertEqual(job.get_status(), JobStatus.FINISHED)
            self.assertEqual(len(ScheduledJobRegistry(queue=queue)), 0)
            self.assertIn(job.id, FinishedJobRegistry(queue=queue, job_class=CustomJob).get_job_ids())

    def test_work_burst_with_queue_class(self):
        """The scheduler enqueues due jobs with the worker's queue_class"""
        queue = CustomQueue(connection=self.connection)
        job = queue.enqueue_at(datetime(2019, 1, 1, tzinfo=timezone.utc), say_hello)

        worker = Worker([queue], connection=self.connection, queue_class=CustomQueue)
        worker.work(burst=True, with_scheduler=True)

        self.assertEqual(job.get_status(refresh=True), JobStatus.FINISHED)
        self.assertEqual(self.connection.llen('rq:queue:default'), 0)

    def test_work_with_serializer(self):
        queue = Queue(connection=self.connection, serializer=JSONSerializer)
        worker = Worker(queues=[queue], connection=self.connection, serializer=JSONSerializer)
        p = Process(target=kill_worker, args=(os.getpid(), False, 5))

        p.start()
        queue.enqueue_at(datetime(2019, 1, 1, tzinfo=timezone.utc), say_hello, meta={'foo': 'bar'})
        worker.work(burst=False, with_scheduler=True)
        p.join(1)
        self.assertIsNotNone(worker.scheduler)
        registry = FinishedJobRegistry(queue=queue)
        self.assertEqual(len(registry), 1)


class TestQueue(RQTestCase):
    def test_enqueue_at(self):
        """queue.enqueue_at() puts job in the scheduled"""
        queue = Queue(connection=self.connection)
        registry = ScheduledJobRegistry(queue=queue)
        scheduler = RQScheduler([queue], connection=self.connection)
        scheduler.acquire_locks()
        # Jobs created using enqueue_at is put in the ScheduledJobRegistry
        job = queue.enqueue_at(datetime(2019, 1, 1, tzinfo=timezone.utc), say_hello)
        self.assertEqual(len(queue), 0)
        self.assertEqual(len(registry), 1)

        # enqueue_at set job status to "scheduled"
        self.assertEqual(job.get_status(), 'scheduled')

        # After enqueue_scheduled_jobs() is called, the registry is empty
        # and job is enqueued
        scheduler.enqueue_scheduled_jobs()
        self.assertEqual(len(queue), 1)
        self.assertEqual(len(registry), 0)

    def test_enqueue_at_with_pipeline(self):
        """queue.enqueue_at() with a pipeline schedules the job when the pipeline is executed"""
        queue = Queue(connection=self.connection)
        registry = ScheduledJobRegistry(queue=queue)
        scheduler = RQScheduler([queue], connection=self.connection)
        scheduler.acquire_locks()

        # Discarding the pipeline writes nothing
        pipeline = self.connection.pipeline()
        job = queue.enqueue_at(datetime(2019, 1, 1, tzinfo=timezone.utc), say_hello, pipeline=pipeline)
        pipeline.reset()
        self.assertFalse(self.connection.exists(job.key))
        self.assertEqual(len(registry), 0)

        # A scheduler run before the pipeline is executed doesn't lose the job
        pipeline = self.connection.pipeline()
        job = queue.enqueue_at(datetime(2019, 1, 1, tzinfo=timezone.utc), say_hello, pipeline=pipeline)
        scheduler.enqueue_scheduled_jobs()
        pipeline.execute()
        self.assertIn(job.id, registry)
        scheduler.enqueue_scheduled_jobs()
        self.assertEqual(queue.job_ids, [job.id])

    def test_enqueue_at_at_front(self):
        """queue.enqueue_at() accepts at_front argument. When true, job will be put at position 0
        of the queue when the time comes for the job to be scheduled"""
        queue = Queue(connection=self.connection)
        registry = ScheduledJobRegistry(queue=queue)
        scheduler = RQScheduler([queue], connection=self.connection)
        scheduler.acquire_locks()
        # Jobs created using enqueue_at is put in the ScheduledJobRegistry
        # job_first should be enqueued first
        job_first = queue.enqueue_at(datetime(2019, 1, 1, tzinfo=timezone.utc), say_hello)
        # job_second will be enqueued second, but "at_front"
        job_second = queue.enqueue_at(datetime(2019, 1, 2, tzinfo=timezone.utc), say_hello, at_front=True)
        self.assertEqual(len(queue), 0)
        self.assertEqual(len(registry), 2)

        # enqueue_at set job status to "scheduled"
        self.assertEqual(job_first.get_status(), 'scheduled')
        self.assertEqual(job_second.get_status(), 'scheduled')

        # After enqueue_scheduled_jobs() is called, the registry is empty
        # and job is enqueued
        scheduler.enqueue_scheduled_jobs()
        self.assertEqual(len(queue), 2)
        self.assertEqual(len(registry), 0)
        self.assertEqual(0, queue.get_job_position(job_second.id))
        self.assertEqual(1, queue.get_job_position(job_first.id))

    def test_enqueue_in(self):
        """queue.enqueue_in() schedules job correctly"""
        queue = Queue(connection=self.connection)
        registry = ScheduledJobRegistry(queue=queue)

        job = queue.enqueue_in(timedelta(seconds=30), say_hello)
        now = datetime.now(timezone.utc)
        scheduled_time = registry.get_scheduled_time(job)
        # Ensure that job is scheduled roughly 30 seconds from now
        self.assertTrue(now + timedelta(seconds=28) < scheduled_time < now + timedelta(seconds=32))

    def test_enqueue_in_with_retry(self):
        """Ensure that the retry parameter is passed
        to the enqueue_at function from enqueue_in.
        """
        queue = Queue(connection=self.connection)
        job = queue.enqueue_in(timedelta(seconds=30), say_hello, retry=Retry(3, [2]))
        self.assertEqual(job.retries_left, 3)
        self.assertEqual(job.retry_intervals, [2])

    def test_custom_connection_pool(self):
        """Connection pool customizing. Ensure that we can properly set a
        custom connection pool class and pass extra arguments"""
        custom_conn = redis.Redis(
            connection_pool=redis.ConnectionPool(
                connection_class=CustomRedisConnection,
                db=4,
                custom_arg='foo',
            )
        )

        queue = Queue(connection=custom_conn)
        scheduler = RQScheduler([queue], connection=custom_conn)

        scheduler_connection = scheduler.connection.connection_pool.get_connection('info')

        self.assertEqual(scheduler_connection.__class__, CustomRedisConnection)
        self.assertEqual(scheduler_connection.get_custom_arg(), 'foo')

    def test_no_custom_connection_pool(self):
        """Connection pool customizing must not interfere if we're using a standard
        connection (non-pooled)"""
        standard_conn = redis.Redis(db=5)

        queue = Queue(connection=standard_conn)
        scheduler = RQScheduler([queue], connection=standard_conn)

        scheduler_connection = scheduler.connection.connection_pool.get_connection('info')

        self.assertEqual(scheduler_connection.__class__, redis.Connection)
