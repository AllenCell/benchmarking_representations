import pandas as pd
import pycytominer
import numpy as np
from src.cellprofiler import utils

def get_metadata():
    meta_path = '/allen/aics/modeling/ritvik/projects/2023_Chandrasekaran_submitted/metadata/platemaps/2020_11_04_CPJUMP1/platemap/JUMP-Target-1_compound_platemap.txt'
    data = pd.read_csv(meta_path, sep="\t", header=None)
    data.columns = data.iloc[0]
    data = data.iloc[1:].reset_index(drop=True)
    data = data.loc[data['well_position'].str.startswith('A')]
    this_path = '/allen/aics/modeling/ritvik/projects/2023_Chandrasekaran_submitted/metadata/external_metadata/JUMP-Target-1_compound_metadata.tsv'
    more_data = pd.read_csv(this_path, sep='\t')
    data = data.merge(more_data, on='broad_sample')

    this_path = '/allen/aics/modeling/ritvik/projects/2023_Chandrasekaran_submitted/benchmark/output/experiment-metadata.tsv'
    data_meta = pd.read_csv(this_path, sep="\t", header=None)
    data_meta.columns = data_meta.iloc[0]
    data_meta = data_meta.iloc[1:].reset_index(drop=True)
    data_meta = data_meta.loc[data_meta['Batch'].isin(['2020_11_04_CPJUMP1'])].reset_index(drop=True)
    return data, data_meta

def agg_norm_singlecells(df_feats, cols):
    data, data_meta = get_metadata()
    all_normalized_df = []
    for plate in df_feats['Assay_Plate_Barcode'].unique():
        test = df_feats.loc[df_feats['Assay_Plate_Barcode'] == plate].reset_index(drop=True)

        aggregate_df = pycytominer.aggregate(
            test,
            strata=['Assay_Plate_Barcode', 'well_position'],
            features=cols,
            operation='median', 
            object_feature='Metadata_ObjectNumber',
        )
        aggregate_df = aggregate_df.merge(data, on='well_position')
        aggregate_df = aggregate_df.merge(data_meta, on='Assay_Plate_Barcode')
        
        normalized_df = pycytominer.normalize(
            profiles=aggregate_df,
            features=cols,
            meta_features=['Assay_Plate_Barcode', 'well_position', 'pert_iname', 'pert_type','Cell_type',
                        'Anomaly', 'control_type', 'Time', 'Density', 'broad_sample',
                        'Perturbation'],
            method="standardize",
            mad_robustize_epsilon=0,
            samples="all"
        )
        normalized_df = pycytominer.normalize(
            profiles=normalized_df,
            features=cols,
            meta_features=['Assay_Plate_Barcode', 'well_position', 'pert_iname', 'pert_type','Cell_type',
                        'Anomaly', 'control_type', 'Time', 'Density', 'broad_sample',
                        'Perturbation'],
            method="standardize",
            samples="control_type == 'negcon'",
        )
        all_normalized_df.append(normalized_df)
    df_final = pd.concat(all_normalized_df, axis=0).reset_index(drop=True)
    df_final['Metadata_broad_sample'] = df_final['broad_sample']
    df_final['Metadata_control_type'] = df_final['control_type']
    df_final['Metadata_Plate'] = df_final['Assay_Plate_Barcode']

    add_metadata = pd.read_csv('/allen/aics/modeling/ritvik/projects/2023_Chandrasekaran_submitted/benchmark/input/JUMP-Target-1_compound_metadata_additional_annotations.tsv', 
                            sep="\t", header=None)
    add_metadata.columns = add_metadata.iloc[0]
    add_metadata = add_metadata.iloc[1:].reset_index(drop=True)
    add_metadata['Metadata_broad_sample'] = add_metadata['broad_sample']
    df_final = df_final.merge(add_metadata[['Metadata_broad_sample', 'target_list']], on='Metadata_broad_sample')
    df_final['Metadata_target_list'] = df_final['target_list']

    plates = ['BR00117017', 'BR00117025', 'BR00116992', 'BR00117024',
        'BR00117015', 'BR00116994', 'BR00116993', 'BR00117050',
        'BR00117016', 'BR00116991', 'BR00117012', 'BR00117019',
        'BR00117009', 'BR00117010', 'BR00117013', 'BR00117026',
        'BR00117011', 'BR00117008', 'BR00116995']
    df_final = df_final.loc[df_final['Assay_Plate_Barcode'].isin(plates)].reset_index(drop=True)

    wells = ['A01', 'A02', 'A09', 'A17', 'A03', 'A04', 'A05', 'A06', 'A07',
        'A08', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A18',
        'A19', 'A20', 'A21', 'A22', 'A23', 'A24']
    df_final = df_final.loc[df_final['well_position'].isin(wells)].reset_index(drop=True)
    df_final['PlateWell'] = df_final['Assay_Plate_Barcode'] + '_' + df_final['well_position']
    return df_final


def run_map_calculation(experiment_df, df_final, get_featuredata, batch_size, null_size, target1_metadata):
    replicability_map_df = pd.DataFrame()
    replicability_fr_df = pd.DataFrame()
    matching_map_df = pd.DataFrame()
    matching_fr_df = pd.DataFrame()
    gene_compound_matching_map_df = pd.DataFrame()
    gene_compound_matching_fr_df = pd.DataFrame()

    replicate_feature = 'Metadata_broad_sample'
    for cell in experiment_df.Cell_type.unique():
        cell_df = experiment_df.query("Cell_type==@cell")
        modality_1_perturbation = "compound"
        modality_1_experiments_df = cell_df.query("Perturbation==@modality_1_perturbation")
        for modality_1_timepoint in modality_1_experiments_df.Time.unique():
            modality_1_timepoint_df = modality_1_experiments_df.query(
                "Time==@modality_1_timepoint"
            )
            modality_1_df = pd.DataFrame()
            for plate in modality_1_timepoint_df.Assay_Plate_Barcode.unique():
                data_df = df_final.loc[df_final['Assay_Plate_Barcode'].isin([plate])]
                data_df = data_df.drop(columns=['Metadata_target_list','target_list']).reset_index(drop=True)
                # data_df = data_df.groupby(['pert_iname']).sample(n=10).reset_index(drop=True)
                modality_1_df = utils.concat_profiles(modality_1_df, data_df)

            # Set Metadata_broad_sample value to "DMSO" for DMSO wells
            modality_1_df[replicate_feature].fillna("DMSO", inplace=True)

            # Remove empty wells
            modality_1_df = utils.remove_empty_wells(modality_1_df)

            # Description
            description = f"{modality_1_perturbation}_{cell}_{utils.time_point(modality_1_perturbation, modality_1_timepoint)}"

            # Calculate replicability mAP
            print(f"Computing {description} replicability")

            modality_1_df["Metadata_negcon"] = np.where(
                modality_1_df["Metadata_control_type"] == "negcon", 1, 0
            )  # Create dummy column

            pos_sameby = ["Metadata_broad_sample"]
            pos_diffby = []
            neg_sameby = ["Metadata_Plate"]
            neg_diffby = ["Metadata_negcon"]

            metadata_df = utils.get_metadata(modality_1_df)
            feature_df = get_featuredata(modality_1_df)
            feature_values = feature_df.values

            result = utils.run_pipeline(
                metadata_df,
                feature_values,
                pos_sameby,
                pos_diffby,
                neg_sameby,
                neg_diffby,
                anti_match=False,
                batch_size=batch_size,
                null_size=null_size,
            )

            result = result.query("Metadata_negcon==0").reset_index(drop=True)

            replicability_map_df, replicability_fr_df = utils.create_replicability_df(
                replicability_map_df,
                replicability_fr_df,
                result,
                pos_sameby,
                0.05,
                modality_1_perturbation,
                cell,
                modality_1_timepoint,
            )

            # Remove DMSO wells
            modality_1_df = utils.remove_negcon_and_empty_wells(modality_1_df)

            # Create consensus profiles
            modality_1_consensus_df = utils.consensus(modality_1_df, replicate_feature)

            # Filter out non-replicable compounds
            # replicable_compounds = list(
            #     replicability_map_df[
            #         (replicability_map_df.Description == description)
            #         & (replicability_map_df.above_q_threshold == True)
            #     ][replicate_feature]
            # )
            replicable_compounds = list(
                replicability_map_df[
                    (replicability_map_df.Description == description)
                ][replicate_feature]
            )
            modality_1_consensus_df = modality_1_consensus_df.query(
                "Metadata_broad_sample==@replicable_compounds"
            ).reset_index(drop=True)
            # Adding additional gene annotation metadata
            modality_1_consensus_df = (
                modality_1_consensus_df.merge(
                    target1_metadata, on="Metadata_broad_sample", how="left"
                )
                .assign(
                    Metadata_matching_target=lambda x: x.Metadata_target_list.str.split("|")
                )
                .drop(["Metadata_target_list"], axis=1)
            )

            # Calculate compound-compound matching
            print(f"Computing {description} matching")

            pos_sameby = ["Metadata_matching_target"]
            pos_diffby = []
            neg_sameby = []
            neg_diffby = ["Metadata_matching_target"]

            metadata_df = utils.get_metadata(modality_1_consensus_df)
            feature_df = utils.get_featuredata(modality_1_consensus_df)
            feature_values = feature_df.values

            result = utils.run_pipeline(
                metadata_df,
                feature_values,
                pos_sameby,
                pos_diffby,
                neg_sameby,
                neg_diffby,
                anti_match=True,
                batch_size=batch_size,
                null_size=null_size,
                multilabel_col="Metadata_matching_target",
            )

            matching_map_df, matching_fr_df = utils.create_matching_df(
                matching_map_df,
                matching_fr_df,
                result,
                pos_sameby,
                0.05,
                modality_1_perturbation,
                cell,
                modality_1_timepoint,
            )

            all_modality_2_experiments_df = cell_df.query(
                "Perturbation!=@modality_1_perturbation"
            )
            for (
                modality_2_perturbation
            ) in all_modality_2_experiments_df.Perturbation.unique():
                modality_2_experiments_df = all_modality_2_experiments_df.query(
                    "Perturbation==@modality_2_perturbation"
                )
                for modality_2_timepoint in modality_2_experiments_df.Time.unique():
                    modality_2_timepoint_df = modality_2_experiments_df.query(
                        "Time==@modality_2_timepoint"
                    )

                    modality_2_df = pd.DataFrame()
                    for plate in modality_2_timepoint_df.Assay_Plate_Barcode.unique():
                        data_df = df_final.loc[df_final['Assay_Plate_Barcode'].isin([plate])]
                        data_df = data_df.drop(columns=['Metadata_target_list','target_list']).reset_index(drop=True)
                        data_df = (
                            data_df.assign(Metadata_modality=modality_2_perturbation)
                            .assign(Metadata_matching_target=lambda x: x.Metadata_gene)
                        )
                        modality_2_df = utils.concat_profiles(modality_2_df, data_df)

                    # Remove empty wells
                    modality_2_df = utils.remove_empty_wells(modality_2_df)

                    # Description
                    description = f"{modality_2_perturbation}_{cell}_{utils.time_point(modality_2_perturbation, modality_2_timepoint)}"

                    # Calculate replicability mAP

                    if not replicability_map_df.Description.str.contains(description).any():
                        print(f"Computing {description} replicability")

                        modality_2_df["Metadata_negcon"] = np.where(
                            modality_2_df["Metadata_control_type"] == "negcon", 1, 0
                        )  # Create dummy column

                        pos_sameby = ["Metadata_broad_sample"]
                        pos_diffby = []
                        neg_sameby = ["Metadata_Plate"]
                        neg_diffby = ["Metadata_negcon"]

                        metadata_df = utils.get_metadata(modality_2_df)
                        feature_df = utils.get_featuredata(modality_2_df)
                        feature_values = feature_df.values

                        result = utils.run_pipeline(
                            metadata_df,
                            feature_values,
                            pos_sameby,
                            pos_diffby,
                            neg_sameby,
                            neg_diffby,
                            anti_match=False,
                            batch_size=batch_size,
                            null_size=null_size,
                        )

                        result = result.query("Metadata_negcon==0").reset_index(drop=True)

                        (
                            replicability_map_df,
                            replicability_fr_df,
                        ) = utils.create_replicability_df(
                            replicability_map_df,
                            replicability_fr_df,
                            result,
                            pos_sameby,
                            0.05,
                            modality_2_perturbation,
                            cell,
                            modality_2_timepoint,
                        )

                    # Remove negcon wells
                    modality_2_df = utils.remove_negcon_and_empty_wells(modality_2_df)

                    # Create consensus profiles
                    modality_2_consensus_df = utils.consensus(
                        modality_2_df, "Metadata_broad_sample"
                    )

                    # Filter out non-replicable genes
                    # replicable_genes = list(
                    #     replicability_map_df[
                    #         (
                    #             replicability_map_df.Description
                    #             == f"{modality_2_perturbation}_{cell}_{utils.time_point(modality_2_perturbation, modality_2_timepoint)}"
                    #         )
                    #         & (replicability_map_df.above_q_threshold == True)
                    #     ][replicate_feature]
                    # )
                    replicable_genes = list(
                        replicability_map_df[
                            (
                                replicability_map_df.Description
                                == f"{modality_2_perturbation}_{cell}_{utils.time_point(modality_2_perturbation, modality_2_timepoint)}"
                            )
                            # & (replicability_map_df.above_q_threshold == True)
                        ][replicate_feature]
                    )
                    modality_2_consensus_df = modality_2_consensus_df.query(
                        "Metadata_broad_sample==@replicable_genes"
                    ).reset_index(drop=True)

                    # Filter out reagents without a sister guide

                    genes_without_sister = (
                        modality_2_consensus_df.Metadata_gene.value_counts()
                        .reset_index()
                        .query("Metadata_gene==1")["index"]
                        .to_list()
                    )

                    modality_2_consensus_for_matching_df = (
                        modality_2_consensus_df.query(
                            "Metadata_gene!=@genes_without_sister"
                        ).reset_index(drop=True)
                    )

                    # Calculate cripsr-crispr matching
                    if modality_2_perturbation == "crispr":
                        if not matching_map_df.Description.str.contains(description).any():
                            print(f"Computing {description} matching")

                            pos_sameby = ["Metadata_matching_target"]
                            pos_diffby = []
                            neg_sameby = []
                            neg_diffby = ["Metadata_matching_target"]

                            metadata_df = utils.get_metadata(modality_2_consensus_for_matching_df)
                            feature_df = utils.get_featuredata(modality_2_consensus_for_matching_df)
                            feature_values = feature_df.values

                            result = utils.run_pipeline(
                                metadata_df,
                                feature_values,
                                pos_sameby,
                                pos_diffby,
                                neg_sameby,
                                neg_diffby,
                                anti_match=False,
                                batch_size=batch_size,
                                null_size=null_size,
                            )

                            matching_map_df, matching_fr_df = utils.create_matching_df(
                                matching_map_df,
                                matching_fr_df,
                                result,
                                pos_sameby,
                                0.05,
                                modality_2_perturbation,
                                cell,
                                modality_2_timepoint,
                            )

                    # Filter out genes that are not perturbed by ORFs or CRISPRs
                    perturbed_genes = list(
                        set(modality_2_consensus_df.Metadata_matching_target)
                    )

                    modality_1_filtered_genes_df = (
                        modality_1_consensus_df[
                            ["Metadata_broad_sample", "Metadata_matching_target"]
                        ]
                        .copy()
                        .explode("Metadata_matching_target")
                        .query("Metadata_matching_target==@perturbed_genes")
                        .reset_index(drop=True)
                        .groupby(["Metadata_broad_sample"])
                        .Metadata_matching_target.apply(list)
                        .reset_index()
                    )

                    modality_1_consensus_filtered_df = modality_1_consensus_df.drop(
                        columns=["Metadata_matching_target"]
                    ).merge(
                        modality_1_filtered_genes_df,
                        on="Metadata_broad_sample",
                        how="inner",
                    )

                    # Calculate gene-compound matching mAP
                    description = f"{modality_1_perturbation}_{cell}_{utils.time_point(modality_1_perturbation, modality_1_timepoint)}-{modality_2_perturbation}_{cell}_{utils.time_point(modality_2_perturbation, modality_2_timepoint)}"
                    print(f"Computing {description} matching")

                    modality_1_modality_2_df = utils.concat_profiles(
                        modality_1_consensus_filtered_df, modality_2_consensus_df
                    )

                    pos_sameby = ["Metadata_matching_target"]
                    pos_diffby = ["Metadata_modality"]
                    neg_sameby = []
                    neg_diffby = ["Metadata_matching_target", "Metadata_modality"]

                    metadata_df = utils.get_metadata(modality_1_modality_2_df)
                    feature_df = utils.get_featuredata(modality_1_modality_2_df)
                    feature_values = feature_df.values

                    result = utils.run_pipeline(
                        metadata_df,
                        feature_values,
                        pos_sameby,
                        pos_diffby,
                        neg_sameby,
                        neg_diffby,
                        anti_match=True,
                        batch_size=batch_size,
                        null_size=null_size,
                        multilabel_col="Metadata_matching_target",
                    )

                    (
                        gene_compound_matching_map_df,
                        gene_compound_matching_fr_df,
                    ) = utils.create_gene_compound_matching_df(
                        gene_compound_matching_map_df,
                        gene_compound_matching_fr_df,
                        result,
                        pos_sameby,
                        0.05,
                        modality_1_perturbation,
                        modality_2_perturbation,
                        cell,
                        modality_1_timepoint,
                        modality_2_timepoint,
                    )
    return replicability_map_df, matching_map_df