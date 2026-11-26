import threading
from time import sleep
from unittest import mock

import pytest

from rq import Queue, SimpleWorker, Worker
from rq.exceptions import NoSuchGroupError
from rq.group import Group
from rq.job import Job, JobStatus
from rq.serializers import JSONSerializer
from rq.utils import as_text
from tests import RQTestCase
from tests.fixtures import say_hello


class TestGroup(RQTestCase):
    job_1_data = Queue.prepare_data(say_hello, job_id='job1')
    job_2_data = Queue.prepare_data(say_hello, job_id='job2')

    def test_get_jobs_uses_group_serializer(self):
        """Group.get_jobs() fetches jobs with the serializer configured on the Group.

        Regression test for https://github.com/rq/rq/issues/2594: get_jobs()
        used to call Job.fetch_many() without a serializer, so jobs enqueued
        with JSONSerializer failed to deserialize on read.
        """
        queue = Queue(connection=self.connection, serializer=JSONSerializer)
        group = Group.create(connection=self.connection, serializer=JSONSerializer)
        job_data = Queue.prepare_data(say_hello, args=('Mo',))
        jobs = group.enqueue_many(queue, [job_data])

        fetched = group.get_jobs()
        self.assertEqual(len(fetched), 1)
        self.assertEqual(fetched[0].id, jobs[0].id)
        self.assertEqual(fetched[0].func_name, 'tests.fixtures.say_hello')

    def test_create_group(self):
        q = Queue(connection=self.connection)
        group = Group.create(connection=self.connection)
        group.enqueue_many(q, [self.job_1_data, self.job_2_data])
        assert isinstance(group, Group)
        assert len(group.get_jobs()) == 2
        q.empty()

    def test_group_cleanup_with_no_jobs(self):
        q = Queue(connection=self.connection)
        group = Group.create(connection=self.connection)
        assert len(group.get_jobs()) == 0
        group.cleanup()
        assert len(group.get_jobs()) == 0
        q.empty()

    def test_group_repr(self):
        group = Group.create(name='foo', connection=self.connection)
        assert group.__repr__() == 'Group(id=foo)'

    def test_group_jobs(self):
        q = Queue(connection=self.connection)
        group = Group.create(connection=self.connection)
        jobs = group.enqueue_many(q, [self.job_1_data, self.job_2_data])
        self.assertCountEqual(group.get_jobs(), jobs)
        q.empty()

    def test_fetch_group(self):
        q = Queue(connection=self.connection)
        enqueued_group = Group.create(connection=self.connection)
        enqueued_group.enqueue_many(q, [self.job_1_data, self.job_2_data])
        fetched_group = Group.fetch(enqueued_group.name, self.connection)
        self.assertCountEqual(enqueued_group.get_jobs(), fetched_group.get_jobs())
        assert len(fetched_group.get_jobs()) == 2
        q.empty()

    def test_add_jobs(self):
        q = Queue(connection=self.connection)
        group = Group.create(connection=self.connection)
        group.enqueue_many(q, [self.job_1_data, self.job_2_data])
        job2 = group.enqueue_many(q, [self.job_1_data, self.job_2_data])[0]
        assert job2 in group.get_jobs()
        self.assertEqual(job2.group_id, group.name)
        q.empty()

    def test_jobs_added_to_group_key(self):
        q = Queue(connection=self.connection)
        group = Group.create(connection=self.connection)
        jobs = group.enqueue_many(q, [self.job_1_data, self.job_2_data])
        job_ids = [job.id for job in group.get_jobs()]
        jobs = list({as_text(job) for job in self.connection.smembers(group.key)})
        self.assertCountEqual(jobs, job_ids)
        q.empty()

    def test_group_id_added_to_jobs(self):
        q = Queue(connection=self.connection)
        group = Group.create(connection=self.connection)
        jobs = group.enqueue_many(q, [self.job_1_data])
        assert jobs[0].group_id == group.name
        fetched_job = Job.fetch(jobs[0].id, connection=self.connection)
        assert fetched_job.group_id == group.name

    def test_group_enqueue_many_with_pipeline_is_invisible_until_commit(self):
        """Group writes buffered on a caller pipeline are invisible to other connections."""
        q = Queue(connection=self.connection)
        group = Group.create(connection=self.connection)
        other_connection = self.connection.__class__(connection_pool=self.connection.connection_pool)

        with self.connection.pipeline() as pipe:
            jobs = group.enqueue_many(q, [self.job_1_data, self.job_2_data], pipeline=pipe)
            self.assertEqual(len(jobs), 2)
            # Nothing is visible before the caller commits
            self.assertFalse(other_connection.exists(group.key))
            self.assertFalse(other_connection.exists(Group.REDIS_GROUP_KEY))
            self.assertFalse(other_connection.exists(Job.key_for('job1')))
            self.assertEqual(other_connection.llen(q.key), 0)
            pipe.execute()

        self.assertTrue(self.connection.exists(group.key))
        self.assertIn(group.name.encode(), self.connection.smembers(Group.REDIS_GROUP_KEY))
        self.assertEqual(q.job_ids, ['job1', 'job2'])
        fetched_group = Group.fetch(group.name, connection=self.connection)
        self.assertEqual({job.id for job in fetched_group.get_jobs()}, {'job1', 'job2'})
        other_connection.close()

    def test_group_enqueue_many_discarded_pipeline_leaves_no_residue(self):
        """Discarding the caller pipeline leaves no queue, jobs or group records."""
        q = Queue(connection=self.connection)
        group = Group.create(connection=self.connection)

        with self.connection.pipeline() as pipe:
            group.enqueue_many(q, [self.job_1_data, self.job_2_data], pipeline=pipe)
            pipe.reset()  # caller gives up the transaction

        self.assertEqual(q.count, 0)
        self.assertFalse(self.connection.exists(group.key))
        self.assertEqual(self.connection.smembers(Group.REDIS_GROUP_KEY), set())
        self.assertFalse(self.connection.exists(Job.key_for('job1')))
        self.assertFalse(self.connection.exists(Job.key_for('job2')))
        with self.assertRaises(NoSuchGroupError):
            Group.fetch(group.name, connection=self.connection)

    def test_group_enqueue_many_with_pipeline_and_dependencies(self):
        """Group jobs inside a caller pipeline honor same-batch and external dependencies."""
        q = Queue(connection=self.connection)
        group = Group.create(connection=self.connection)
        finished_parent = Job.create(func=say_hello, connection=self.connection)
        finished_parent.save()
        finished_parent.set_status(JobStatus.FINISHED)
        finished_parent.save()

        job_datas = [
            Queue.prepare_data(say_hello, job_id='g-first'),
            Queue.prepare_data(say_hello, job_id='g-second', depends_on='g-first'),
            Queue.prepare_data(say_hello, job_id='g-third', depends_on=finished_parent),
        ]
        with self.connection.pipeline() as pipe:
            jobs = group.enqueue_many(q, job_datas, pipeline=pipe)
            pipe.execute()

        by_id = {job.id: job for job in jobs}
        self.assertEqual(by_id['g-first'].get_status(refresh=False), JobStatus.QUEUED)
        self.assertEqual(by_id['g-second'].get_status(refresh=False), JobStatus.DEFERRED)
        self.assertEqual(by_id['g-third'].get_status(refresh=False), JobStatus.QUEUED)
        self.assertEqual(q.job_ids, ['g-first', 'g-third'])
        self.assertEqual(q.deferred_job_registry.get_job_ids(), ['g-second'])
        # Every job, including the deferred one, is registered with the group
        self.assertEqual(
            {as_text(job_id) for job_id in self.connection.smembers(group.key)},
            {'g-first', 'g-second', 'g-third'},
        )

    def test_group_enqueue_many_with_pipeline_worker_unlocks_chain(self):
        """A worker unlocks grouped follow-up jobs registered through a caller pipeline."""
        q = Queue(connection=self.connection)
        group = Group.create(connection=self.connection)
        worker = SimpleWorker([q], connection=self.connection)
        job_datas = [
            Queue.prepare_data(say_hello, job_id='g-chain-first'),
            Queue.prepare_data(say_hello, job_id='g-chain-second', depends_on='g-chain-first'),
        ]

        with self.connection.pipeline() as pipe:
            group.enqueue_many(q, job_datas, pipeline=pipe)
            pipe.execute()

        self.assertEqual(q.job_ids, ['g-chain-first'])
        worker.work(burst=True)
        self.assertEqual(Job.fetch('g-chain-first', connection=self.connection).get_status(), JobStatus.FINISHED)
        self.assertEqual(Job.fetch('g-chain-second', connection=self.connection).get_status(), JobStatus.FINISHED)
        self.assertEqual(q.deferred_job_registry.get_job_ids(), [])

    def test_deleted_jobs_removed_from_group(self):
        q = Queue(connection=self.connection)
        group = Group.create(connection=self.connection)
        group.enqueue_many(q, [self.job_1_data, self.job_2_data])
        job = group.get_jobs()[0]
        job.delete()
        group.cleanup()
        redis_jobs = list({as_text(job) for job in self.connection.smembers(group.key)})
        assert job.id not in redis_jobs
        assert job not in group.get_jobs()

    def test_cleanup_checks_alive_jobs_with_one_exists_call(self):
        q = Queue(connection=self.connection)
        group = Group.create(connection=self.connection)
        group.enqueue_many(q, [self.job_1_data, self.job_2_data])
        with mock.patch.object(self.connection, 'exists', wraps=self.connection.exists) as exists:
            group.cleanup()
        exists.assert_called_once()
        self.assertEqual(len(exists.call_args.args), 2)
        self.assertEqual(len(group.get_jobs()), 2)

    def test_group_added_to_registry(self):
        q = Queue(connection=self.connection)
        group = Group.create(connection=self.connection)
        group.enqueue_many(q, [self.job_1_data])
        redis_groups = {as_text(group) for group in self.connection.smembers('rq:groups')}
        assert group.name in redis_groups
        q.empty()

    @pytest.mark.slow
    def test_expired_jobs_removed_from_group(self):
        q = Queue(connection=self.connection)
        w = SimpleWorker([q], connection=q.connection)
        short_lived_job = Queue.prepare_data(say_hello, result_ttl=1)
        group = Group.create(connection=self.connection)
        group.enqueue_many(q, [short_lived_job, self.job_1_data])
        w.work(burst=True, max_jobs=1)
        sleep(2)
        self.connection.delete(Group.REDIS_GROUP_CLEANING_KEY)
        w.run_maintenance_tasks()
        group.cleanup()
        assert len(group.get_jobs()) == 1
        assert self.job_1_data.job_id in [job.id for job in group.get_jobs()]
        q.empty()

    @pytest.mark.slow
    def test_empty_group_removed_from_group_list(self):
        q = Queue(connection=self.connection)
        w = SimpleWorker([q], connection=q.connection)
        short_lived_job = Queue.prepare_data(say_hello, result_ttl=1)
        group = Group.create(connection=self.connection)
        group.enqueue_many(q, [short_lived_job])
        w.work(burst=True, max_jobs=1)
        sleep(2)
        self.connection.delete(Group.REDIS_GROUP_CLEANING_KEY)
        w.run_maintenance_tasks()
        redis_groups = {as_text(group) for group in self.connection.smembers('rq:groups')}
        assert group.name not in redis_groups

    @pytest.mark.slow
    def test_fetch_expired_group_raises_error(self):
        q = Queue(connection=self.connection)
        w = SimpleWorker([q], connection=q.connection)
        short_lived_job = Queue.prepare_data(say_hello, result_ttl=1)
        group = Group.create(connection=self.connection)
        group.enqueue_many(q, [short_lived_job])
        w.work(burst=True, max_jobs=1)
        sleep(2)
        self.connection.delete(Group.REDIS_GROUP_CLEANING_KEY)
        w.run_maintenance_tasks()
        self.assertRaises(NoSuchGroupError, Group.fetch, group.name, group.connection)
        q.empty()

    def test_get_group_key(self):
        group = Group(name='foo', connection=self.connection)
        self.assertEqual(Group.get_key(group.name), 'rq:group:foo')

    def test_all_returns_all_groups(self):
        q = Queue(connection=self.connection)
        group1 = Group.create(name='group1', connection=self.connection)
        Group.create(name='group2', connection=self.connection)
        group1.enqueue_many(q, [self.job_1_data, self.job_2_data])
        all_groups = Group.all(self.connection)
        assert len(all_groups) == 1
        assert 'group1' in [group.name for group in all_groups]
        assert 'group2' not in [group.name for group in all_groups]

    def test_all_deletes_missing_groups(self):
        q = Queue(connection=self.connection)
        group = Group.create(connection=self.connection)
        jobs = group.enqueue_many(q, [self.job_1_data])
        jobs[0].delete()
        assert not self.connection.exists(Group.get_key(group.name))
        assert Group.all(connection=self.connection) == []

    def test_group_lock_expires_before_next_maintenance(self):
        q = Queue(connection=self.connection)
        worker = Worker([q], connection=self.connection, maintenance_interval=30)
        worker.clean_group_registries()
        ttl = self.connection.ttl(Group.REDIS_GROUP_CLEANING_KEY)
        self.assertTrue(0 < ttl < worker.maintenance_interval)

    def test_group_lock_with_float_maintenance_interval(self):
        q = Queue(connection=self.connection)
        worker = Worker([q], connection=self.connection, maintenance_interval=30.0)
        worker.clean_group_registries()
        self.assertEqual(self.connection.ttl(Group.REDIS_GROUP_CLEANING_KEY), 29)

    def test_group_lock_with_zero_maintenance_interval(self):
        q = Queue(connection=self.connection)
        worker = Worker([q], connection=self.connection, maintenance_interval=0)
        worker.clean_group_registries()
        self.assertTrue(0 < self.connection.pttl(Group.REDIS_GROUP_CLEANING_KEY) <= 1000)

    def test_clean_group_registries_locks_across_workers(self):
        q = Queue(connection=self.connection)
        worker = Worker([q], connection=self.connection)

        def expired_group():
            group = Group.create(connection=self.connection)
            jobs = group.enqueue_many(q, [Queue.prepare_data(say_hello)])
            self.connection.delete(jobs[0].key)
            return group

        first = expired_group()
        worker.clean_group_registries()
        self.assertFalse(self.connection.sismember(Group.REDIS_GROUP_KEY, first.name))
        self.assertTrue(self.connection.exists(Group.REDIS_GROUP_CLEANING_KEY))

        second = expired_group()
        worker.clean_group_registries()
        self.assertTrue(self.connection.sismember(Group.REDIS_GROUP_KEY, second.name))

        self.connection.delete(Group.REDIS_GROUP_CLEANING_KEY)
        worker.clean_group_registries()
        self.assertFalse(self.connection.sismember(Group.REDIS_GROUP_KEY, second.name))
        q.empty()

    def test_only_one_worker_cleans_groups_per_window(self):
        q = Queue(connection=self.connection)
        workers = [Worker([q], connection=self.connection) for _ in range(20)]

        with mock.patch.object(Group, 'clean_registries') as clean:
            threads = [threading.Thread(target=w.clean_group_registries) for w in workers]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(clean.call_count, 1)

            self.connection.delete(Group.REDIS_GROUP_CLEANING_KEY)
            workers[0].clean_group_registries()
            self.assertEqual(clean.call_count, 2)

    def test_group_cleaning_key_expires(self):
        q = Queue(connection=self.connection)
        Worker([q], connection=self.connection).clean_group_registries()
        ttl = self.connection.ttl(Group.REDIS_GROUP_CLEANING_KEY)
        self.assertTrue(0 < ttl <= 899)
