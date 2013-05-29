"""
Migrate management command.
"""

from __future__ import print_function

from copy import deepcopy
import os
from itertools import chain
from optparse import make_option
import pprint
import subprocess
from StringIO import StringIO

from two2three.pgen2 import driver
from two2three import pygram, pytree

from django.core.management.base import BaseCommand
from django.utils.importlib import import_module

from south.migration import Migrations

from south_rebase.diff import ModelsDiff


class Command(BaseCommand):
    option_list = BaseCommand.option_list + (
        make_option('--git', action='store_const', dest='plan_generator',
                    const='git', default='interactive',
                    help='Rebase all migrations from last commit'),
    )

    def handle(self, app_name=None, **options):
        self.options = options

        if app_name is None:
            print('No app_label given')
            return

        migrations = Migrations(app_name)
        rebase_plan = self.split_migrations(migrations)
        if not rebase_plan.migrations:
            print('No migrations to rebase')
            return

        diffs = []
        for plan_elem in rebase_plan.migrations:
            """Prepare diffs"""
            diffs.append(self.diff_frozen_models(plan_elem['previous'],
                                                 plan_elem['migration']))

        prev_migration = rebase_plan.onto
        self.prepare_file_cache()
        for diff, migration in zip(
                diffs, (m['migration'] for m in rebase_plan.migrations)):
            """Rebase - rename migrations and apply diffs"""
            # Prepare new name for migration
            last_migration_no, _ = split_migration_name(prev_migration.name())
            _, migration_base_name = split_migration_name(migration.name())

            migration_path = self.find_migration_path(migration)
            migration_dir = os.path.dirname(migration_path)

            new_migration_filename = '%04i_%s.py' % (last_migration_no + 1,
                                                     migration_base_name)
            new_migration_path = os.path.join(
                migration_dir, new_migration_filename)

            # patch migration tree
            new_migration_tree = self.get_updated_migration(
                migration, prev_migration, diff=diff)

            self.write_file(new_migration_path,
                            unicode(new_migration_tree).encode('utf-8'))
            self.remove_file(migration_path)

            prev_migration = migration

        self.flush_file_cache()

    def write_file(self, path, content):
        self.files_to_write[path] = content

    def remove_file(self, path):
        self.files_to_remove.append(path)

    def prepare_file_cache(self):
        self.files_to_remove = []
        self.files_to_write = {}  # (filename, content)

    def flush_file_cache(self):
        if all(map(os.path.exists, self.files_to_remove)) and \
                not any(map(os.path.exists, self.files_to_write)):
            for path in self.files_to_remove:
                os.remove(path)
            for path, data in self.files_to_write.items():
                if not isinstance(data, str):
                    data = data.encode('utf-8')
                with open(path, 'w') as f:
                    f.write(data)

    def find_migration_path(self, migration):
        """Find path to file this migration comes from"""
        path = import_module(migration.migration_class().__module__).__file__
        if path.endswith('.pyc'):
            path = path[:-1]
        return path

    def split_migrations(self, all_migrations):
        """Pick splitting algorithm and return rebase_plan"""
        generators = dict(interactive=self.interactive_split_migrations,
                          git=self.git_split_migrations)
        return generators[self.options['plan_generator']](all_migrations)

    def interactive_split_migrations(self, all_migrations):
        """Prepare rebase_plan by asking user about conflicting migrations"""
        rebase_plan = RebasePlan()

        known_migrations_by_number = {}
        for m in all_migrations:
            num, base_name = split_migration_name(m.name())
            known_migrations_by_number.setdefault(num, [])
            known_migrations_by_number[num].append(m)

        rebase_plan.migrations = []
        for num, num_migrations in known_migrations_by_number.items():
            """For every migration number, ask about conflicts"""
            if len(num_migrations) > 1:
                selected_migration = interactive_select(
                    [(m.name(), m) for m in num_migrations],
                    'Colliding migration names:',
                    'pick the one to rebase')

                last_num_migrations = set(known_migrations_by_number[num - 1])
                """last_num_migrations are migrations for previous number,
                minus the one that is already selected for rebasing"""
                if rebase_plan.migrations:
                    last_num_migrations.discard(
                        rebase_plan.migrations[-1]['migration'])

                prev = only_element(last_num_migrations)
                rebase_plan.migrations.append({
                    'migration': selected_migration,
                    'previous': prev,
                })

        last_num_migrations = set(num_migrations)
        if rebase_plan.migrations:
            last_num_migrations.discard(
                rebase_plan.migrations[-1]['migration'])
        rebase_plan.onto = only_element(last_num_migrations)

        return rebase_plan

    def git_split_migrations(self, all_migrations):
        """return plan for rebasing all migrations from last commit"""
        repo_path = os.path.abspath(git_cdup())
        new_migration_paths = set()

        normalize = lambda p: os.path.realpath(p)

        for added, deleted, path in git_diff_tree():
            if added and '/migrations/' in path:
                if not deleted:
                    new_migration_paths.add(normalize(
                        os.path.join(repo_path, path)))
                else:
                    print(u'WARNING: Not using migration "%s" because '
                          u'it is modified and not new' % path)

        # Pick which migrations to rebase
        to_rebase = []
        other_migrations = []
        for migration in all_migrations:
            normalized = normalize(self.find_migration_path(migration))
            if normalized in new_migration_paths:
                to_rebase.append(migration)
            else:
                other_migrations.append(migration)

        first_num = min(migration_number(m) for m in to_rebase)

        others_last_num = max(migration_number(m)
                              for m in other_migrations)

        prev_num = first_num - 1
        last_common_migration = only_element(
            m for m in other_migrations if migration_number(m) == prev_num)

        rebase_plan = RebasePlan()
        for prev, migration in zip(chain([last_common_migration], to_rebase),
                                   to_rebase):
            rebase_plan.migrations.append({
                'previous': prev,
                'migration': migration,
            })
        rebase_plan.onto = only_element(
            m for m in other_migrations
            if migration_number(m) == others_last_num)

        return rebase_plan

    def diff_frozen_models(self, prev_migration, migration):
        """Return differences between frozen models"""
        app_label = migration.app_label()

        prev_models = prev_migration.migration_class().models
        models = migration.migration_class().models

        models_for_diff = {
            m for m in chain(prev_models.keys(), models.keys())
            if m.startswith('%s.' % app_label)}

        filter_for_diff = lambda ms: {k: v for k, v in ms.items()
                                      if k in models_for_diff}

        diff = ModelsDiff.diff(filter_for_diff(prev_models),
                               filter_for_diff(models))
        return diff

    def get_updated_migration(self, migration, prev_migration, diff):
        """Return tree of migration with diff applied"""
        self.driver = driver.Driver(pygram.python_grammar,
                                    convert=pytree.convert)
        migration_src = open(self.find_migration_path(migration)).read() + '\n'
        tree = self.driver.parse_string(migration_src)
        patched_tree = deepcopy(tree)
        self.patch_migration_tree(
            patched_tree,
            diff.apply(prev_migration.migration_class().models))
        return patched_tree

    def patch_migration_tree(self, tree, new_models):
        model_dict_assignment = find_assignments(
            find_classes(tree, name='Migration').next(),
            name='models').next()
        new_models_assignment_tree = self.driver.parse_string(
            'models = %s\n' % pprint_str(new_models))
        new_models_assignment = find_assignments(
            new_models_assignment_tree).next()
        model_dict_assignment.children[2] = new_models_assignment.children[2]


class RebasePlan(object):
    """Object representing a plan for rebasing."""
    def __init__(self):
        self.onto = None  # What is the new base migration?
        self.migrations = []
        """List of dicts of migrations to rebase (in target order). Used keys:
            - migration - the migration that needs rebasing
            - previous - a migration this one was based on.
        """


def split_migration_name(name):
    num, base = name.split('_', 1)
    return int(num), base


def migration_number(migration):
    return split_migration_name(migration.name())[0]


def iterate_tree(tree):
    yield tree
    for child in tree.children:
        for node in iterate_tree(child):
            yield node


def find_classes(tree, name=None):
    """Find all classes in tree"""
    for node in iterate_tree(tree):
        if node.type == pygram.python_symbols.classdef:
            if name is None or node.children[1].value == name:
                yield node


def find_assignments(tree, name=None):
    for node in iterate_tree(tree):
        if node.type == pygram.python_symbols.expr_stmt and \
                (name is None or name == node.children[0].value) and \
                node.children[1].value == '=':
            yield node


def interactive_select(options, pre_message=None, post_message=None):
    """Select one of options interactively"""
    if len(options) == 0:
        return
    if not isinstance(options[0], tuple):
        options = [(unicode(o), o) for o in options]
    if len(options) == 1:
        return options[0][1]

    def print_options():
        if pre_message is not None:
            print(pre_message)

        for i, (opt_name, opt_value) in enumerate(options):
            print('% 4i: %s' % (i, opt_name))

        if post_message is not None:
            print(post_message)

    option_index = -1

    while not 0 <= option_index < len(options):
        try:
            print_options()
            option_index = int(raw_input())
        except ValueError:
            pass

    return options[option_index][1]


def only_element(collection):
    """Return the only element of collection, or throw if it has more"""
    coll_iter = iter(collection)
    try:
        elem = coll_iter.next()
    except StopIteration:
        raise ValueError('Empty collection')
    try:
        coll_iter.next()
        raise ValueError('Collection has more elements')
    except StopIteration:
        return elem


def pprint_str(obj, **kwargs):
    assert 'stream' not in kwargs
    stream = StringIO()
    pprint.pprint(obj, stream=stream, **kwargs)
    return stream.getvalue()


def git_diff_tree(treeish='HEAD', other_treeish=None):
    """Does git diff-tree and yields tuples of (added, deleted, file_path)"""
    args = 'git diff-tree --numstat -r -z'.split() + [treeish]
    if other_treeish is not None:
        args.append(other_treeish)
    raw_output = subprocess.check_output(args)
    # Drop the first line - it's the commit hash
    lines = raw_output.split(chr(0))[1:]
    for line in filter(None, lines):
        added, deleted, path = tuple(line.split('\t'))
        yield (int(added), int(deleted), path.decode('utf-8'))

def git_cdup():
    """Return path to root of git project"""
    return subprocess.check_output('git rev-parse --show-cdup'.split()).strip()
