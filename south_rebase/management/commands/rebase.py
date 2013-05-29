"""
Migrate management command.
"""

from __future__ import print_function

from copy import deepcopy
import os
from itertools import chain
import pprint
from StringIO import StringIO

from two2three.pgen2 import driver
from two2three import pygram, pytree

from django.core.management.base import BaseCommand
from django.utils.importlib import import_module

from south.migration import Migrations

from south_rebase.diff import ModelsDiff


class Command(BaseCommand):
    def handle(self, app_name=None, **options):
        files_to_remove = []
        files_to_write = {}  # (filename, content)

        migrations = Migrations(app_name)
        last_common, other_branch = self.split_migrations(migrations)
        if not other_branch:
            print('No migrations to rebase')
            return

        diffs = []
        for prev_migration, migration in zip(
                [last_common] + other_branch, other_branch):
            """Prepare diffs"""
            diffs.append(self.diff_frozen_models(prev_migration, migration))

        for diff, prev_migration, migration in zip(
                diffs, [migrations[-1]] + other_branch, other_branch):
            """Rebase - rename migrations and apply diffs"""
            last_migration_no, _ = split_migration_name(migrations[-1].name())
            _, migration_base_name = split_migration_name(migration.name())

            migration_path = self.find_migration_path(migration)
            migration_dir = os.path.dirname(migration_path)

            new_migration_tree = self.get_updated_migration(
                migration, prev_migration, diff=diff)
            new_migration_filename = '%04i_%s.py' % (last_migration_no + 1,
                                                     migration_base_name)
            new_migration_path = os.path.join(
                migration_dir, new_migration_filename)
            files_to_write[new_migration_path] = unicode(new_migration_tree)
            files_to_remove.append(migration_path)

        if all(map(os.path.exists, files_to_remove)) and \
                not any(map(os.path.exists, files_to_write)):
            for path in files_to_remove:
                os.remove(path)
            for path, data in files_to_write.items():
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

    def split_migrations(self, migrations):
        """Split migrations that need rebase out of the migrations list"""
        known_migrations_by_number = {}
        for m in migrations:
            num, base_name = split_migration_name(m.name())
            known_migrations_by_number.setdefault(num, [])
            known_migrations_by_number[num].append(m)

        other_migrations = []
        last_common = None
        for num, num_migrations in known_migrations_by_number.items():
            if len(num_migrations) > 1:
                print('Colliding migration names:')
                for i, m in enumerate(num_migrations):
                    print('% 4i: %s' % (i, m.name()))
                print('pick the one to rebase')
                rebase_i = int(raw_input())
                other_migrations.append(num_migrations[rebase_i])
                del num_migrations[rebase_i]
                if last_common is None:
                    last_common = known_migrations_by_number[num - 1][0]

        for m in other_migrations:
            migrations.remove(m)

        return last_common, other_migrations

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


def split_migration_name(name):
    num, base = name.split('_', 1)
    return int(num), base



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


def pprint_str(obj, **kwargs):
    assert 'stream' not in kwargs
    stream = StringIO()
    pprint.pprint(obj, stream=stream, **kwargs)
    return stream.getvalue()
