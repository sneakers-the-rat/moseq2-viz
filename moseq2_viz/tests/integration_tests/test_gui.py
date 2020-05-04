import os
import shutil
import ruamel.yaml as yaml
from unittest import TestCase
from moseq2_viz.gui import get_groups_command, add_group_by_session, add_group_by_subject, make_crowd_movies_command,\
                plot_usages_command, plot_scalar_summary_command, plot_transition_graph_command, \
                plot_syllable_durations_command

class TestGUI(TestCase):

    def test_get_groups_command(self):
        index_path = 'data/test_index.yaml'

        with open(index_path, 'r') as f:
            index_data = yaml.safe_load(f)
        f.close()

        groups, uuids = [], []
        subjectNames, sessionNames = [], []
        for f in index_data['files']:
            if f['uuid'] not in uuids:
                uuids.append(f['uuid'])
                groups.append(f['group'])
                subjectNames.append(f['metadata']['SubjectName'])
                sessionNames.append(f['metadata']['SessionName'])

        num_groups = get_groups_command(index_path)

        assert num_groups == len(set(groups)), "Number of returned groups is incorrect"

    def test_add_group_by_session(self):
        index_path = 'data/test_index.yaml'
        tmp_yaml = 'data/tmp_copy.yaml'

        with open(index_path, 'r') as f:
            index_data = yaml.safe_load(f)
        f.close()

        with open(tmp_yaml, 'w') as g:
            yaml.safe_dump(index_data, g)


        value = 'blackStockOFA80GritSanded_012517'
        group = 'test1'
        exact = False
        lowercase = False
        negative = False

        add_group_by_session(tmp_yaml, value, group, exact, lowercase, negative)

        assert not os.path.samefile(index_path, tmp_yaml), "Index file was not updated."
        os.remove(tmp_yaml)


    def test_add_group_by_subject(self):
        index_path = 'data/test_index.yaml'
        tmp_yaml = 'data/tmp_copy.yaml'

        with open(index_path, 'r') as f:
            index_data = yaml.safe_load(f)
        f.close()

        with open(tmp_yaml, 'w') as g:
            yaml.safe_dump(index_data, g)

        value = '012517'
        group = 'test1'
        exact = False
        lowercase = False
        negative = False

        add_group_by_subject(tmp_yaml, value, group, exact, lowercase, negative)

        assert not os.path.samefile(index_path, tmp_yaml), "Index file was not updated"
        os.remove(tmp_yaml)

    def test_plot_usages_command(self):
        gen_dir = 'data/gen_plots/'
        index_file = 'data/test_index.yaml'
        model_path = 'data/test_model.p'
        sort = True
        count = 'usage'
        max_syllable = 40
        group = ('default', 'Group1')
        output_file = gen_dir+'test_usages'

        plot_usages_command(index_file, model_path, sort, count, max_syllable, group, output_file)

        assert (os.path.exists(gen_dir + 'test_usages.png')), "Usages PNG plot was not saved"
        assert (os.path.exists(gen_dir + 'test_usages.pdf')), "Usages PDF plot was not saved"
        os.remove(gen_dir + 'test_usages.png')
        os.remove(gen_dir + 'test_usages.pdf')
        os.removedirs(gen_dir)

    def test_plot_scalar_summary_command(self):
        index_file = 'data/test_index.yaml'
        gen_dir = 'data/gen_plots/'
        output_file = gen_dir + 'test_scalar'

        df = plot_scalar_summary_command(index_file, output_file)
        assert not df.empty, "Scalar DataFrame was not return correctly; is empty."
        assert (os.path.exists(gen_dir + 'test_scalar_position.png')), "Position Summary PNG plot was not saved"
        assert (os.path.exists(gen_dir + 'test_scalar_position.pdf')), "Position Summary PDF plot was not saved"
        assert (os.path.exists(gen_dir + 'test_scalar_summary.png')), "Scalar Summary PNG plot was not saved"
        assert (os.path.exists(gen_dir + 'test_scalar_summary.pdf')), "Scalar Summary PDF plot was not saved"
        shutil.rmtree(gen_dir)

    def test_plot_transition_graph_command(self):
        gen_dir = 'data/gen_plots/'
        index_file = 'data/test_index.yaml'
        model_path = 'data/test_model.p'
        config_file = 'data/config.yaml'
        max_syllable = 40
        group = ('Group1', 'default')
        output_file = gen_dir+'test_transitions'

        plot_transition_graph_command(index_file, model_path, config_file, max_syllable, group, output_file)
        assert (os.path.exists(output_file + '.png')), "Transition PNG graph was not saved"
        assert (os.path.exists(output_file + '.pdf')), "Transition PDF graph was not saved"
        shutil.rmtree(gen_dir)


    def test_plot_syllable_durations_command(self):
        gen_dir = 'data/gen_plots/'
        index_file = 'data/test_index.yaml'
        model_path = 'data/test_model.p'
        group = ('Group1', 'default')
        count = 'usage'
        max_syllable = 40
        output_file = gen_dir + 'test_durations'

        plot_syllable_durations_command(model_path, index_file, group, count, max_syllable, output_file)
        assert (os.path.exists(output_file + '.png')), "Durations PNG graph was not saved"
        assert (os.path.exists(output_file + '.pdf')), "Durations PNG graph was not saved"
        shutil.rmtree(gen_dir)

    def test_copy_h5_metadata_to_yaml_command(self):
        print()