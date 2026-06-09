import argparse
import os
import warnings
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import AutoMinorLocator
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_mutual_info_score


DEFAULT_DATAPATH = (
    "/nfs/research/jlees/vrbouza/data/clustering_benchmarking/"
    "2025_09_04_morepangenomesims"
)
DEFAULT_FONT_REGULAR = (
    "/nfs/research/jlees/vrbouza/data/ibm-plex-sans/fonts/complete/ttf/"
    "IBMPlexSans-Regular.ttf"
)
DEFAULT_FONT_ITALIC = (
    "/nfs/research/jlees/vrbouza/data/ibm-plex-sans/fonts/complete/ttf/"
    "IBMPlexSans-Italic.ttf"
)
DEFAULT_FONT_BOLD = (
    "/nfs/research/jlees/vrbouza/data/ibm-plex-sans/fonts/complete/ttf/"
    "IBMPlexSans-Bold.ttf"
)
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_SEEDS = str(PROJECT_ROOT / "data" / "random_numbers.txt")

CLUSTERERS = ["cdhit", "mmseqs2"]
SEQTYPES = ["nt", "aa"]
PARAMORDER = ["st", "c"]
DEFAULT_PARAMS = {"st": "nt", "c": 0.9}
AXIS_TITLE_FONT_SIZE = 12
DOPREM = True

FANCYDICT = {
    "cdhit/nt": "CD-HIT (NT)",
    "mmseqs2/nt": "MMseqs2 (NT)",
    "cdhit/aa": "CD-HIT (AA)",
    "mmseqs2/aa": "MMseqs2 (AA)",
}

CONFIGDICT = {
    "adj_rand_index": {
        "ylabel": "Adjusted Rand index (adim.)",
        "ylimits": (0.8, 1.0),
        "ylimits_c": (0.5, 1.0),
    },
    "purity": {"ylabel": "Purity (adim.)", "ylimits": (0.85, 1.0)},
    "adj_mutual_info": {
        "ylabel": "Adjusted mutual information (adim.)",
        "ylimits": (0.8, 1.0),
    },
    "v_measure": {"ylabel": "V-measure (adim.)", "ylimits": (0.8, 1.0)},
    "runtime": {"ylabel": "Runtime (s)"},
}

CONFIGDICT_COLOURS = {
    "cdhit/nt": "#D41645",
    "cdhit/aa": "#E58F9E",
    "mmseqs2/nt": "#193F90",
    "mmseqs2/aa": "#8BB8E8",
}


def nicesp(uglysp):
    return " ".join(uglysp.split(".")).capitalize()


def get_font_properties(args):
    return (
        FontProperties(fname=args.font_regular),
        FontProperties(fname=args.font_italic),
        FontProperties(fname=args.font_bold),
    )


def get_param_dict_from_splits(thesplits):
    outdict = {}
    for split in thesplits:
        subsplits = split.split("-")
        if subsplits[1].replace(".", "").isdigit():
            outdict[subsplits[0]] = float(subsplits[1])
        else:
            outdict[subsplits[0]] = subsplits[1]
    return outdict


def get_labels_list_from_df(indf):
    labels = []
    for column in indf.columns:
        tmp = indf[indf[column] >= 0.0].index
        if len(tmp) > 1:
            print(tmp)
        labels.append(tmp[0])
    return labels


def get_purity(inlab, truthdf):
    nclusters = len(set(inlab))
    sumofmaxes = 0
    for column in truthdf.columns:
        countlist = [0] * nclusters
        for gene in truthdf[truthdf[column] == True].index:
            countlist[inlab[int(gene.split("_")[1])]] += 1
        sumofmaxes += max(countlist)
    return float(sumofmaxes) / float(len(truthdf.index))


def calculate_values_from_cluster_matrix(infotuple, indf, truthlab, truthdf):
    probelab = get_labels_list_from_df(indf)
    outlist = [
        True,
        infotuple[0],
        infotuple[1],
        infotuple[2],
        metrics.adjusted_rand_score(truthlab, probelab),
        get_purity(probelab, truthdf),
        adjusted_mutual_info_score(truthlab, probelab).item(),
    ]
    outlist += [el.item() for el in metrics.homogeneity_completeness_v_measure(truthlab, probelab)]
    return outlist


def get_df_from_clusterer(clusterer, folderpath):
    if clusterer == "cdhit":
        listoflists = []
        setofgenes = set()
        listofclusters = []
        tmpdict = {}
        with open(os.path.join(folderpath, "cdhit.clstr"), "r") as f:
            tmpclusterid = -1
            for line in f:
                if line[0] == ">":
                    tmpclusterid = int(line.replace(">", "").split(" ")[1].strip())
                    tmpdict[tmpclusterid] = {}
                    listofclusters.append(tmpclusterid)
                else:
                    tmpgeneid = line.strip().split(">")[1].split("...")[0]
                    setofgenes.add(tmpgeneid)
                    tmpdict[tmpclusterid][tmpgeneid] = (
                        float(line.strip().split("at")[1].strip().replace("%", "")) / 100.0
                    ) if "*" not in line else 2.0

        listofgenes = list(setofgenes)
        listofgenes.sort(key=lambda x: int(x.split("_")[1]))
        for cluster in listofclusters:
            row = [cluster]
            for gene in listofgenes:
                row.append(tmpdict[cluster][gene] if gene in tmpdict[cluster] else -1.0)
            listoflists.append(row)
        outdf = pd.DataFrame(listoflists, columns=["cluster_id"] + listofgenes)
        return outdf.set_index("cluster_id")

    if clusterer == "mmseqs2":
        firstdf = pd.read_csv(
            os.path.join(folderpath, "mmseqs2_cluster.tsv"),
            names=["cluster_id", "gene_id"],
            sep="\t",
        )
        clusterlist = list(set([int(iC.split("_")[1]) for iC in firstdf["cluster_id"]]))
        clusterlist.sort()
        genelist = list(set(list(firstdf["gene_id"])))
        genelist.sort(key=lambda x: int(x.split("_")[1]))

        listoflists = []
        for cluster_index in range(len(clusterlist)):
            tmpset = set(
                firstdf[
                    firstdf["cluster_id"] == "geneid_" + str(clusterlist[cluster_index])
                ]["gene_id"]
            )
            row = [cluster_index]
            for gene in genelist:
                row.append(+1.0 if gene in tmpset else -1.0)
            listoflists.append(row)

        outdf = pd.DataFrame(listoflists, columns=["cluster_id"] + genelist)
        return outdf.set_index("cluster_id")

    raise RuntimeError("Clusterer " + clusterer + " not supported!")


def get_time_diff_from_file(inpath):
    time0 = None
    time1 = None
    with open(inpath, "r") as f:
        for line in f:
            if "=>" in line:
                splits = line.strip().split("=>")
                time0 = datetime.strptime(splits[0], "%d/%m/%Y-%H:%M:%S")
                time1 = datetime.strptime(splits[1], "%d/%m/%Y-%H:%M:%S")
                break
    return (time1 - time0).total_seconds()


def get_species_name(inpath):
    with open(inpath, "r") as f:
        for line in f:
            return line.strip()
    return ""


def check_status_of_folder(clusterer, path):
    if clusterer == "cdhit":
        filenam = "cdhit.clstr"
    elif clusterer == "mmseqs2":
        filenam = "mmseqs2_cluster.tsv"
    else:
        print("Invalid clusterer " + clusterer)
        return False
    checkpath = os.path.join(path, filenam)
    if os.path.isfile(checkpath):
        return True
    print("Invalid file " + checkpath)
    return False


def get_info_from_folder(theargs):
    thedir, theass, theseed, datapath = theargs
    truthpath = [
        os.path.join(datapath, "simulations", str(theass), str(theseed), el)
        for el in os.listdir(os.path.join(datapath, "simulations", str(theass), str(theseed)))
        if "truth_matrix" in el
    ][0]
    truthmatrix = pd.read_csv(truthpath, sep="\t")
    truthmatrix = truthmatrix.set_index("gene_id")
    truthlabels = list(truthmatrix["original_gene"])
    one_hot = pd.get_dummies(truthmatrix["original_gene"])
    truthdf = truthmatrix.drop("original_gene", axis=1)
    truthdf = truthdf.join(one_hot)
    print(f"\t- Getting information from {thedir} execution, {theass} assembly, and {theseed} seed")

    speciesfile = os.path.join(thedir, str(theass), "assembly_species.txt")
    if os.path.isfile(speciesfile):
        nameofass = get_species_name(speciesfile)
    else:
        warnings.warn(
            f"Missing {speciesfile}; falling back to assembly ID {theass}",
            RuntimeWarning,
            stacklevel=2,
        )
        nameofass = str(theass)

    listoflists = []
    for folder_name in os.listdir(os.path.join(thedir, str(theass), str(theseed))):
        splits = folder_name.split("_")
        tmpclusterer = splits[0]
        paramdict = get_param_dict_from_splits(splits[1:]) if len(splits) > 1 else {}

        for key in paramdict:
            if key not in DEFAULT_PARAMS:
                raise RuntimeError(
                    "Parameter "
                    + str(key)
                    + " from folder "
                    + os.path.join(thedir, folder_name)
                    + " is not contemplated."
                )

        folderpath = os.path.join(thedir, str(theass), str(theseed), folder_name)
        if not check_status_of_folder(tmpclusterer, folderpath):
            continue

        thedf = get_df_from_clusterer(tmpclusterer, folderpath)
        runtime = get_time_diff_from_file(os.path.join(folderpath, "timebenchmark.txt"))
        paramlist = [paramdict[el] if el in paramdict else DEFAULT_PARAMS[el] for el in PARAMORDER]
        listoflists.append(
            calculate_values_from_cluster_matrix(
                (theass, theseed, tmpclusterer), thedf, truthlabels, truthdf
            )
            + paramlist
            + [runtime]
        )
    return (listoflists, theass, nameofass)


def plotter(theargs):
    name, datadf, namedict, outfolder, assembly, datatype, font_props = theargs
    ibmplexsans, ibmplexsansitalics, ibmplexsansbold = font_props
    print(f"\t- Plotting c-plot {name} for simulations of {namedict[assembly]}")
    subdf = datadf[
        (datadf.index.get_level_values("assembly") == assembly)
        & (datadf.index.get_level_values("simulations") == (datatype == "simulations"))
    ]

    fig = plt.figure(1, dpi=150)
    ax = fig.subplots()
    xs = list(set(list(subdf["c"].astype(float))))
    xs.sort()
    clusterers = list(set(list(subdf.index.get_level_values("clusterer"))))

    for seqtype in SEQTYPES:
        availcl = list(set(list(subdf[subdf["st"] == seqtype].index.get_level_values("clusterer"))))
        ynams = [el + "/" + seqtype for el in clusterers if el in availcl]
        ymean = np.zeros((len(ynams), len(xs)))
        ystd = np.zeros((len(ynams), len(xs)))

        for cluster_index, ynam in enumerate(ynams):
            for x_index, x_value in enumerate(xs):
                tmpdf = subdf[
                    (subdf["st"] == seqtype)
                    & (subdf.index.get_level_values("simulations") == True)
                    & (subdf.index.get_level_values("assembly") == assembly)
                    & (subdf.index.get_level_values("clusterer") == ynam.split("/")[0])
                    & (subdf["c"] == x_value)
                ][name].astype(float)
                ymean[cluster_index, x_index] = tmpdf.mean()
                ystd[cluster_index, x_index] = tmpdf.std()

        for y_index, ynam in enumerate(ynams):
            ax.plot(xs, ymean[y_index, :], "-", c=CONFIGDICT_COLOURS[ynam], label=FANCYDICT[ynam])
            ax.fill_between(
                xs,
                [max(ymean[y_index, j] - ystd[y_index, j], 0.0) for j in range(len(ymean[y_index, :]))],
                ymean[y_index, :] + ystd[y_index, :],
                alpha=0.5,
                edgecolor=CONFIGDICT_COLOURS[ynam],
                facecolor=CONFIGDICT_COLOURS[ynam],
            )

    if name in CONFIGDICT and "ylimits_c" in CONFIGDICT[name]:
        ax.set_ylim(CONFIGDICT[name]["ylimits_c"][0], CONFIGDICT[name]["ylimits_c"][1])
    elif name in CONFIGDICT and "ylimits" in CONFIGDICT[name]:
        ax.set_ylim(CONFIGDICT[name]["ylimits"][0], CONFIGDICT[name]["ylimits"][1])
    else:
        ax.set_ylim(0.0, None)

    ax.set_xlabel("c (adim.)", fontproperties=ibmplexsans, loc="right", fontsize=AXIS_TITLE_FONT_SIZE)
    ax.set_ylabel(
        CONFIGDICT[name]["ylabel"] if name in CONFIGDICT and "ylabel" in CONFIGDICT[name] else name,
        fontproperties=ibmplexsans,
        loc="top",
        fontsize=AXIS_TITLE_FONT_SIZE,
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="major", direction="in")
    ax.tick_params(which="minor", direction="in")
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.ticklabel_format(useMathText=True)
    ax.get_yaxis().get_offset_text().set_x(-0.075)
    ax.get_yaxis().get_offset_text().set_fontproperties(ibmplexsans)
    ax.set_xlim(xs[0], xs[-1])

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(ibmplexsans)

    plt.text(0, 1.01, namedict[assembly], fontproperties=ibmplexsansitalics, horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)
    plt.text(1, 1.01, "Simulations", fontproperties=ibmplexsans, horizontalalignment="right", verticalalignment="bottom", transform=ax.transAxes)
    if DOPREM:
        plt.text(0.5, 1.01, "Preliminary", fontproperties=ibmplexsansbold, horizontalalignment="center", verticalalignment="bottom", transform=ax.transAxes)
    plt.legend(loc="best", frameon=False, prop=ibmplexsans, handlelength=0.5, handletextpad=0.75, labelspacing=0.3)

    outnamescaff = name.replace(" ", "").replace("#", "NumberOf")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            os.path.join(outfolder, "_".join(["plot_c", datatype, assembly, outnamescaff]) + "." + ext),
            bbox_inches="tight",
        )
    fig.clf()
    del fig, ax


def plotter_pointplots(theargs):
    name, datadf, namedict, outfolder, assembly, datatype, font_props = theargs
    ibmplexsans, ibmplexsansitalics, ibmplexsansbold = font_props
    print(f"\t- Plotting point-plot {name} for simulations of {namedict[assembly]}")
    subdf = datadf[
        (datadf.index.get_level_values("assembly") == assembly)
        & (datadf.index.get_level_values("simulations") == (datatype == "simulations"))
        & (datadf["c"] == DEFAULT_PARAMS["c"])
    ]
    clusterers = list(set(list(subdf.index.get_level_values("clusterer"))))
    x = []
    for clusterer, seqtype in [(c, st) for st in SEQTYPES for c in clusterers]:
        if len(list(subdf[(subdf.index.get_level_values("clusterer") == clusterer) & (subdf["st"] == seqtype)][name])):
            x.append(clusterer + "/" + seqtype)

    x_fancy = [FANCYDICT[value] for value in x]
    ymean = []
    ystd = []
    for x_value in x:
        tmpdf = subdf[
            (subdf.index.get_level_values("simulations") == True)
            & (subdf.index.get_level_values("assembly") == assembly)
            & (subdf.index.get_level_values("clusterer") == x_value.split("/")[0])
            & (subdf["st"] == x_value.split("/")[1])
        ][name].astype(float)
        ymean.append(tmpdf.mean())
        ystd.append(tmpdf.std())

    fig = plt.figure(1, dpi=150)
    ax = fig.subplots()
    for index in range(len(x)):
        ax.errorbar(
            x_fancy[index],
            ymean[index],
            c=CONFIGDICT_COLOURS[x[index]],
            yerr=[[ystd[index] if ymean[index] > ystd[index] else ymean[index]], [ystd[index]]],
            capsize=4.0,
            fmt="o",
        )

    if name in CONFIGDICT and "ylimits" in CONFIGDICT[name]:
        ax.set_ylim(CONFIGDICT[name]["ylimits"][0], CONFIGDICT[name]["ylimits"][1])
    else:
        ax.set_ylim(0.0, None)

    ax.set_xlabel("Clusterer", fontproperties=ibmplexsans, loc="right", fontsize=AXIS_TITLE_FONT_SIZE)
    ax.set_ylabel(
        CONFIGDICT[name]["ylabel"] if name in CONFIGDICT and "ylabel" in CONFIGDICT[name] else name,
        fontproperties=ibmplexsans,
        loc="top",
        fontsize=AXIS_TITLE_FONT_SIZE,
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="major", direction="in")
    ax.tick_params(which="minor", direction="in")
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.get_yaxis().get_offset_text().set_x(-0.075)
    ax.get_yaxis().get_offset_text().set_fontproperties(ibmplexsans)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(ibmplexsans)

    plt.text(0, 1.01, namedict[assembly], fontproperties=ibmplexsansitalics, horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)
    plt.text(1, 1.01, "Simulations", fontproperties=ibmplexsans, horizontalalignment="right", verticalalignment="bottom", transform=ax.transAxes)
    if DOPREM:
        plt.text(0.5, 1.01, "Preliminary", fontproperties=ibmplexsansbold, horizontalalignment="center", verticalalignment="bottom", transform=ax.transAxes)

    outnamescaff = name.replace(" ", "").replace("#", "NumberOf")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            os.path.join(outfolder, "_".join(["plot_point", datatype, assembly, outnamescaff]) + "." + ext),
            bbox_inches="tight",
        )
    fig.clf()
    del fig, ax


def load_seeds(seedsfile):
    seeds = []
    with open(seedsfile, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                seeds.append(int(stripped))
    seeds.sort()
    return seeds


def build_results_dataframe(listoflists):
    outdf = pd.DataFrame(
        listoflists,
        columns=[
            "simulations",
            "assembly",
            "seed",
            "clusterer",
            "adj_rand_index",
            "purity",
            "adj_mutual_info",
            "homogeneity",
            "completeness",
            "v_measure",
        ]
        + PARAMORDER
        + ["runtime"],
    )
    index = pd.MultiIndex.from_frame(outdf[["simulations", "assembly", "seed", "clusterer"]])
    outdf = outdf.drop(["simulations", "assembly", "seed", "clusterer"], axis=1)
    return outdf.set_index(index)


def main():
    parser = argparse.ArgumentParser(description="Analyse gene clustering benchmark runs.")
    parser.add_argument("runfolder", default="./")
    parser.add_argument("--out-folder", dest="outfolder", default="./temp_runanalysis")
    parser.add_argument("--nthreads", "-j", type=int, default=1)
    parser.add_argument("--datapath", default=DEFAULT_DATAPATH)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--font-regular", default=DEFAULT_FONT_REGULAR)
    parser.add_argument("--font-italic", default=DEFAULT_FONT_ITALIC)
    parser.add_argument("--font-bold", default=DEFAULT_FONT_BOLD)
    args = parser.parse_args()

    plt.rcParams.update({"figure.max_open_warning": 0})
    font_props = get_font_properties(args)
    seedsfile = args.seeds

    lsdirs = next(os.walk(args.runfolder))[1]
    if "simulations" not in lsdirs:
        raise RuntimeError("No valid folders found!")
    simassemblies = next(os.walk(os.path.join(args.runfolder, "simulations")))[1]

    print("> Getting seeds")
    seeds = load_seeds(seedsfile)
    print("> Got {} seeds".format(len(seeds)))

    print("\n> Getting info...")
    gettinginfotasks = [
        (os.path.join(args.runfolder, "simulations"), assembly, seed, args.datapath)
        for assembly in simassemblies
        for seed in seeds
    ]

    listoflists = []
    namedict = {}
    if args.nthreads <= 1:
        for task in gettinginfotasks:
            tmpout = get_info_from_folder(task)
            listoflists += tmpout[0]
            namedict[tmpout[1]] = tmpout[2]
    else:
        pool = Pool(args.nthreads)
        for result in pool.map(get_info_from_folder, gettinginfotasks):
            listoflists += result[0]
            namedict[result[1]] = result[2]
        pool.close()
        pool.join()
    print("\n> Done!")

    outdf = build_results_dataframe(listoflists)
    assemblies = set(list(outdf.index.get_level_values("assembly")))

    if not os.path.isdir(args.outfolder):
        os.makedirs(args.outfolder)

    plotstodo = ["adj_rand_index", "purity", "adj_mutual_info", "v_measure", "runtime"]
    print("\n> Preparing plotting tasks...")
    plottingtasks_pointplots = [
        (plot_name, outdf, namedict, args.outfolder, assembly, "simulations", font_props)
        for plot_name in plotstodo
        for assembly in assemblies
    ]
    plottingtasks = [
        (plot_name, outdf, namedict, args.outfolder, assembly, "simulations", font_props)
        for plot_name in plotstodo
        for assembly in assemblies
    ]

    print("\n> Plotting...")
    if args.nthreads <= 1:
        for task in plottingtasks_pointplots:
            plotter_pointplots(task)
        for task in plottingtasks:
            plotter(task)
    else:
        pool = Pool(args.nthreads)
        pool.map(plotter_pointplots, plottingtasks_pointplots)
        pool.map(plotter, plottingtasks)
        pool.close()
        pool.join()

    print("\n> Done!\n")


if __name__ == "__main__":
    main()
