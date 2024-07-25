use std::{
    cmp::max,
    collections::HashSet,
    fs::File,
    io::{self, BufRead},
    path::{Path, PathBuf},
    sync::Arc,
    thread,
};

use anyhow::anyhow;
use clap::Parser;
use poppy::{
    utils::{benchmark, time_it_once, ByteSize, Stats},
    BloomFilter, OptLevel, Params, DEFAULT_VERSION,
};
use poppy_filters as poppy;
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use statrs::statistics::Statistics;

// When built with musl, the allocator slows down a lot insertion
// so we have to use jemalloc to get back to the expected performance
#[cfg(all(target_env = "musl", target_pointer_width = "64"))]
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[derive(Debug, Parser)]
pub struct Args {
    /// Verbose output
    #[clap(short, long)]
    verbose: bool,
    /// The number of jobs to use when parallelization is possible. Set
    /// to 0 to take the optimal number of jobs (may be the amount of CPUs).
    /// For write operations the original filter is copied into the memory
    /// of each job so you can expect the memory of the whole process to
    /// be N times the size of the filter.
    #[clap(short, long, default_value = "2")]
    jobs: usize,
    /// Poppy commands
    #[clap(subcommand)]
    command: Command,
}

#[derive(Debug, Parser)]
enum Command {
    /// Create a new bloom filter
    Create(Create),
    /// Insert data into an existing bloom filter
    Insert(Insert),
    /// Checks entries against an existing bloom filter
    Check(Check),
    /// Benchmark the bloom filter. If the bloom filter
    /// behaves in an unexpected way, the benchmark fails.
    /// Input data is read from stdin.
    Bench(Benchmark),
    /// Show information about an existing bloom filter
    Show(Show),
}

#[derive(Debug, Parser)]
struct Create {
    /// Creates a specific version of Bloom Filter
    /// Set to 1 if cross-compatibility with other DCSO libraries
    #[clap(long, default_value_t=DEFAULT_VERSION)]
    version: u8,
    /// Capacity of the bloom filter. This value is ignored if
    /// the filter is created from a dataset.
    #[clap(short, long, default_value = "10000")]
    capacity: usize,
    /// False positive rate
    #[clap(short, long, default_value = "0.01")]
    probability: f64,
    /// Optimize the bloom filter.
    /// 0 for none,
    /// 1 for space,
    /// 2 for speed,
    /// 3 for best
    #[clap(short = 'O', long = "opt-lvl", default_value_t = 0)]
    optimize: u8,
    /// File to store the bloom filter
    file: PathBuf,
    /// Fills the filter with the dataset. Each file must contains
    /// an entry per line.
    dataset: Vec<String>,
}

#[derive(Debug, Parser)]
struct Insert {
    /// Reads input from stdin (useful when poppy also insert from files)
    #[clap(long)]
    stdin: bool,
    /// File containing the bloom filter to update
    file: String,
    /// Input files containing one entry per line, if no files is specified
    /// entries to insert are read from stdin
    inputs: Vec<String>,
}

#[derive(Debug, Parser)]
struct Check {
    /// checks input from stdin
    #[clap(long)]
    stdin: bool,
    /// do not print anything to stdout
    #[clap(long)]
    silent: bool,
    /// Only show entries not in the bloom filter
    #[clap(long)]
    verify: bool,
    /// File containing the bloom filter
    file: String,
    /// Input files containing one entry per line, if no files is specified
    /// entries to check are read from stdin
    inputs: Vec<String>,
}

#[derive(Debug, Parser)]
struct Benchmark {
    /// Size (in MB) limiting the number of entries to use for
    /// the benchmark. Set it to 0 for full dataset size.
    #[clap(short, long, default_value_t = 100)]
    size_mb: usize,
    /// Limit the number of entries in the test set.
    /// Not used to limit the number of entries in the filter.
    #[clap(long)]
    test_size: Option<usize>,
    /// Number of runs to compute statistics
    #[clap(short, long, default_value_t = 50)]
    runs: u32,
    /// Allowed tolerance (in percentage) on false positive proba before raising error
    #[clap(short, long, default_value_t = 0.2)]
    fp_tol: f64,
    /// Version of filter to use (used only when no file is specified)
    #[clap(short,long, default_value_t=DEFAULT_VERSION)]
    version: u8,
    /// False positive probability to use (used only when no file is specified)
    #[clap(long, default_value_t = 0.001)]
    fpp: f64,
    /// Optimization level to use (used only when no file is specified)
    #[clap(short = 'O', long, default_value_t = 0)]
    opt: u8,
    /// File containing the bloom filter. If no file is provided a filter will be
    /// created from the input data and benchmark will be done on it.
    file: Option<String>,
}

#[derive(Debug, Parser)]
struct Show {
    /// File containing the bloom filter
    file: String,
}

fn show_bf_properties(b: &BloomFilter) {
    println!(
        "\tversion                                  : {}",
        b.version()
    );
    println!(
        "\tcapacity (desired number of elements)    : {}",
        b.capacity()
    );
    println!(
        "\tn (estimated number of elements)         : {}",
        b.count_estimate()
    );
    println!("\tfpp (false positive probability)         : {}", b.fpp());

    println!(
        "\tLength of data                           : {}",
        ByteSize::from_bytes(b.data().len())
    );
    println!(
        "\tSize of bloom filter                     : {}",
        ByteSize::from_bytes(b.size_in_bytes())
    );
}

#[inline(always)]
fn optimal_jobs(jobs: usize) -> usize {
    match jobs {
        0 => thread::available_parallelism()
            .map(|j| j.get())
            .unwrap_or(1),
        _ => jobs,
    }
}

fn count_lines_parallel<P: AsRef<Path> + Clone>(
    files: &[P],
    jobs: usize,
) -> Result<usize, anyhow::Error> {
    let jobs = optimal_jobs(jobs);
    let mut count = 0usize;

    let batches = files.chunks(max(files.len() / jobs, 1));
    let mut handles = vec![];

    for batch in batches {
        let batch: Vec<PathBuf> = batch.iter().map(|p| p.as_ref().to_path_buf()).collect();

        handles.push(thread::spawn(move || {
            let mut count = 0usize;
            for f in batch {
                let of = File::open(f)?;
                count += io::BufReader::new(of).lines().count();
            }
            Ok::<_, anyhow::Error>(count)
        }));
    }

    for h in handles {
        count += h.join().expect("failed to join thread")?;
    }

    Ok(count)
}

fn process_file(bf: &mut BloomFilter, input: &String, verbose: bool) -> Result<(), anyhow::Error> {
    if verbose {
        eprintln!("processing file: {input}");
    }
    let in_file = std::fs::File::open(input)?;

    for line in std::io::BufReader::new(in_file).lines() {
        let line = line?;
        bf.insert_bytes(&line)?;
    }

    Ok(())
}

fn parallel_insert(
    bf: BloomFilter,
    files: Vec<String>,
    jobs: usize,
    verbose: bool,
) -> Result<BloomFilter, anyhow::Error> {
    if files.is_empty() {
        return Ok(bf);
    }

    let jobs = optimal_jobs(jobs);

    let batches = files.chunks(max(files.len() / jobs, 1));
    let mut bfs = vec![];
    for _ in 0..(batches.len().saturating_sub(1)) {
        bfs.push(bf.clone())
    }
    // we move bf to prevent doing an additional copy
    bfs.push(bf);

    let mut handles = vec![];

    // map processing
    for batch in batches {
        let batch: Vec<String> = batch.to_vec();
        let mut copy = bfs.pop().unwrap();

        let h = thread::spawn(move || {
            for input in batch {
                process_file(&mut copy, &input, verbose)?;
            }

            Ok::<_, anyhow::Error>(copy)
        });
        handles.push(h)
    }

    // reduce
    let mut bfs = handles
        .into_iter()
        .map(|h| h.join().expect("failed to join thread").unwrap())
        .collect::<Vec<BloomFilter>>();

    // code should never panic here
    let mut out = bfs.pop().expect("bpfs must always have one item");
    while let Some(bf) = bfs.pop() {
        out.union_merge(&bf)?;
    }

    Ok(out)
}

fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();

    match args.command {
        Command::Create(o) => {
            // if the output file exists, we check it is a valid bloom filter file
            // this prevents writting an unwanted file from the dataset if we forgot
            // one cli argument
            if o.file.exists() {
                BloomFilter::from_reader(File::open(&o.file)?)
                    .map_err(|_| anyhow!("output file exists but it is not a valid poppy file"))?;
            }

            let capacity = match o.dataset.len() {
                0 => o.capacity,
                _ => {
                    if args.verbose {
                        eprintln!("counting lines in dataset")
                    }
                    count_lines_parallel(&o.dataset, args.jobs)?
                }
            };

            // we create bloom filter
            let b = {
                let p = Params::new(capacity, o.probability)
                    .version(o.version)
                    .opt(OptLevel::try_from(o.optimize)?);
                BloomFilter::try_from(p)?
            };

            // we insert and save
            let bf = parallel_insert(b, o.dataset, args.jobs, args.verbose)?;
            let mut f = File::create(o.file)?;
            bf.write(&mut f)?;
        }
        Command::Insert(o) => {
            let bloom_file = File::open(&o.file)?;
            let mut bf = BloomFilter::from_reader(bloom_file)?;

            // if we pipe in entries via stdin
            if o.stdin || o.inputs.is_empty() {
                for line in std::io::BufReader::new(io::stdin()).lines() {
                    bf.insert_bytes(line?)?;
                }
            }

            // we insert and save
            let bf = parallel_insert(bf, o.inputs, args.jobs, args.verbose)?;
            let mut output = File::create(o.file)?;
            bf.write(&mut output)?;
        }

        Command::Check(o) => {
            let bloom_file = File::open(&o.file)?;
            let bf = Arc::new(std::sync::RwLock::new(BloomFilter::from_reader(
                bloom_file,
            )?));

            if o.stdin || o.inputs.is_empty() {
                let t = time_it_once(|| {
                    let bf = bf.read().unwrap();
                    for line in io::stdin().lines() {
                        let line = line.unwrap();
                        let ok = bf.contains_bytes(&line);

                        if o.verify {
                            if !ok {
                                println!("{line}: NOK")
                            }
                            continue;
                        }

                        if ok && !o.silent {
                            println!("{line}")
                        }
                    }
                });

                eprintln!("Check time: {t:?}");
            }

            if !o.inputs.is_empty() {
                let mut handles = vec![];
                let files = o.inputs.clone();

                let batches = files.chunks(max(files.len() / args.jobs, 1));

                for batch in batches {
                    let shared = Arc::clone(&bf);
                    let batch: Vec<String> = batch.to_vec();

                    let h = thread::spawn(move || {
                        let bf = shared
                            .read()
                            .map_err(|e| anyhow!("failed to lock mutex: {}", e))?;

                        for input in batch {
                            let in_file = std::fs::File::open(&input)?;

                            for line in std::io::BufReader::new(in_file).lines() {
                                let line = line?;
                                let ok = bf.contains_bytes(&line);

                                if o.verify {
                                    if !ok {
                                        println!("{line}: NOK")
                                    }
                                    continue;
                                }

                                if ok && !o.silent {
                                    println!("{line}")
                                }
                            }
                        }

                        Ok::<(), anyhow::Error>(())
                    });
                    handles.push(h)
                }

                for h in handles {
                    h.join().expect("failed to join thread")?;
                }
            }
        }
        Command::Bench(o) => {
            let mut rng: StdRng = SeedableRng::from_seed([42; 32]);
            let mut entries = vec![];
            let mut marked = HashSet::new();
            let mut dataset_size = 0usize;
            let mut queries_per_seconds = vec![];
            let limit = match o.size_mb {
                0 => ByteSize::from_mb(usize::MAX),
                _ => ByteSize::from_mb(o.size_mb),
            };

            println!("Preparing dataset");
            for line in io::stdin().lines() {
                let line = line?;
                let h = wyhash::wyhash(line.as_bytes(), 42);

                if ByteSize::from_bytes(dataset_size) > limit {
                    break;
                }

                if !marked.contains(&h) {
                    dataset_size += line.as_bytes().len();
                    marked.insert(h);
                    entries.push(line);
                }
            }

            let b = match o.file {
                Some(f) => {
                    println!("Benchmarking filter: {}", f);
                    let b = BloomFilter::from_reader(File::open(&f)?)?;
                    for entry in &entries {
                        if !b.contains_bytes(entry) {
                            return Err(anyhow!(
                                "bloom filter to benchmark must contain all entries"
                            ));
                        };
                    }
                    b
                }
                None => {
                    println!("Benchmarking dataset:");
                    let p = Params::new(entries.len(), o.fpp)
                        .version(o.version)
                        .opt(o.opt.try_into()?);
                    let mut b = BloomFilter::try_from(p)?;
                    b.fill(entries.iter().collect::<Vec<&String>>())?;
                    b
                }
            };

            show_bf_properties(&b);

            let test_len = o.test_size.unwrap_or(entries.len());

            println!("\nQuery performance:");
            println!(
                "\tdataset size: {} ({}MB)",
                ByteSize::from_bytes(dataset_size),
                ByteSize::from_bytes(dataset_size).in_mb().round()
            );
            println!("\ttest size: {test_len}");
            println!();

            // shuffling entry dataset
            entries.shuffle(&mut rng);

            let mut fpps = vec![];
            for mut_prob in (10..=100).step_by(10) {
                let mut tmp = entries
                    .iter()
                    .take(test_len)
                    .cloned()
                    .collect::<Vec<String>>();

                let mutated_lines = tmp
                    .iter_mut()
                    .map(|e| {
                        let mut mutated = false;
                        if rng.gen_range(0..=100) < mut_prob {
                            mutated = true;
                            unsafe { e.as_bytes_mut() }
                                .iter_mut()
                                .for_each(|b| *b ^= rng.gen_range(0..=255));
                        }
                        (mutated, e.clone())
                    })
                    .collect::<Vec<(bool, String)>>();

                let mut stats = Stats::new();
                let test_size_bytes = mutated_lines.iter().map(|(_m, l)| l.len()).sum();

                // we compute real fpp (no need to benchmark this)
                mutated_lines.iter().for_each(|(m, l)| {
                    let is_in_bf = b.contains_bytes(l);
                    // data has been mutated
                    if *m {
                        if is_in_bf {
                            stats.inc_fp()
                        } else {
                            stats.inc_tn()
                        }
                    }
                });

                // we compute query duration
                let query_dur = benchmark(
                    || {
                        mutated_lines.iter().for_each(|(_, l)| {
                            b.contains_bytes(l);
                        })
                    },
                    o.runs,
                );

                let qps = mutated_lines.len() as f64 / query_dur.as_secs_f64();
                println!(
                    "\tcondition: {}% of queried values are in filter",
                    100 - mut_prob
                );
                println!("\tquery duration: {:?}", query_dur);
                println!(
                    "\tquery speed: {:.1} queries/s -> {:.1} MB/s",
                    qps,
                    ByteSize::from_bytes(test_size_bytes).in_mb() / query_dur.as_secs_f64()
                );
                println!("\tfp rate = {:3}", stats.fp_rate());
                println!();

                fpps.push(stats.fp_rate());
                queries_per_seconds.push(qps);
            }

            let avg_fpp: f64 = fpps.mean();

            if !avg_fpp.is_nan() && avg_fpp > b.fpp() + b.fpp() * o.fp_tol {
                return Err(anyhow!(
                    "bloom filter fp rate is not respected: real={} VS expected={}",
                    avg_fpp,
                    b.fpp()
                ));
            }

            let avg_qps = queries_per_seconds.mean().round();
            println!("Average queries per seconds: {avg_qps}")
        }
        Command::Show(o) => {
            let bloom_file = File::open(&o.file)?;
            let b = BloomFilter::partial_from_reader(bloom_file)?;
            println!("File: {}", o.file);
            show_bf_properties(&b);
        }
    }
    Ok(())
}
