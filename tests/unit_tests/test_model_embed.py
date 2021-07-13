import os
import unittest
from os.path import exists
from unittest import TestCase
from moseq2_viz.util import parse_index
from moseq2_viz.scalars.util import scalars_to_dataframe
from moseq2_viz.model.util import compute_behavioral_statistics
from moseq2_viz.model.embed import get_Xy_values, run_2d_embedding, run_2d_scalar_embedding

class TestModelEmbed(TestCase):

    def setUp(self):

        index_path = 'data/test_index.yaml'
        model_path = 'data/test_model.p'

        _, self.sorted_index = parse_index(index_path)

        # compute session scalar data
        self.scalar_df = scalars_to_dataframe(self.sorted_index, model_path=model_path)

        # compute syllable usage and scalar statistics
        self.mean_df = compute_behavioral_statistics(self.scalar_df, count='usage',
                                                     groupby=['group', 'uuid'],
                                                     usage_normalization=True)

    def test_run_2d_embedding(self):

        stat = 'usage'
        output_file = 'data/test_embedding.pdf'
        embedding = 'PCA'

        run_2d_embedding(self.mean_df, stat=stat, output_file=output_file, embedding=embedding)

        assert exists(output_file)
        os.remove(output_file)

    def test_run_2d_scalar_embedding(self):
        output_file = '2d_scalar_embedding.pdf'
        embedding = 'PCA'
        n_components = 2

        run_2d_scalar_embedding(self.scalar_df, output_file=output_file, embedding=embedding, n_components=n_components)

        assert exists(output_file)
        os.remove(output_file)

if __name__ == '__main__':
    unittest.main()
